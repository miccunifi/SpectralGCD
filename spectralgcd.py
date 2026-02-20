import argparse
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from utils.general_utils import AverageMeter, init_experiment
from utils.cluster_and_log_utils import log_accs_from_preds
from model import DINOHead, info_nce_logits, SupConLoss, DistillLoss, ContrastiveLearningViewGenerator

import torch.nn.functional as F

from clip.clip_utils import load_clip_model, setup_trainable_parameters

from utils.general_utils import set_wandb, extract_dictionary_text_features, extract_student_dictionary_text_features, CustomCosineAnnealingLR

import open_clip
import random




def compute_zero_shot_teacher_rep(dictionary_text_features, tmp_dataloader, model_clip, logger,
                                  path_to_saved_cross_modal_representations, dataset_name, device,
                                  save_zero_shot_rep="yes", temperature=0.01):
    """Compute cross-modal representations for every training sample using the frozen CLIP teacher.

    For each image, computes the cosine similarity against all dictionary text features,
    scaled by 1/temperature. Results are cached to disk when save_zero_shot_rep='yes' and
    path_to_saved_cross_modal_representations is set, to avoid recomputation across runs.

    Args:
        dictionary_text_features: Text features for filtered concepts [K, D], produced by
            extract_dictionary_text_features. Will be L2-normalized internally.
        tmp_dataloader: DataLoader over the training set (no augmentation, no drop_last).
            Each batch yields (images, labels, unique_indices, mask).
        model_clip: Frozen CLIP teacher model used to encode images.
        logger: Logger instance for info messages.
        path_to_saved_cross_modal_representations: Directory path where cached representations
            are stored/loaded. If empty string, caching is disabled.
        dataset_name: Dataset name (e.g., 'cub', 'scars') used to construct the cache filename.
        device: Torch device for computation.
        save_zero_shot_rep: If 'yes', save computed representations to disk for caching.
            Cached files must be deleted manually when the dictionary or teacher model changes.
        temperature: Temperature scaling applied to cosine similarities (default: 0.01).
            Must match the temperature used for the student in train() for KL distillation
            to be meaningful.

    Returns:
        Dict mapping sample unique-index (int) -> cross-modal representation (np.ndarray of shape [K]).
    """
    model_clip.eval()


    path_to_rep = None

    if path_to_saved_cross_modal_representations and path_to_saved_cross_modal_representations != '':
        path_to_rep = os.path.join(path_to_saved_cross_modal_representations, f"{dataset_name}_zs_rep.pt")
        if os.path.exists(path_to_rep):
            logger.info(f"Loading saved zero-shot representations from {path_to_rep}")
            saved_cross_modal_rep = torch.load(path_to_rep, weights_only=False)
            return saved_cross_modal_rep

    saved_cross_modal_rep = {}

    norm_dictionary_text_features = dictionary_text_features / dictionary_text_features.norm(dim=-1, keepdim=True)

    # compute similarity between image features and text features
    for idx_batch, batch in enumerate(tqdm(tmp_dataloader)):
        images, _, uqidxs, _ = batch
        images = images.cuda(non_blocking=True, device=device)
        with torch.no_grad():
            feats = model_clip.encode_image(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)

            zs_rep = feats @ norm_dictionary_text_features.T
            # NOTE: 0.01 temperature scaling is applied, both to student and teacher since they use the same scale
            zs_rep = zs_rep / temperature

            # Save representations using uqidxs
            for i, uqidx in enumerate(uqidxs):
                saved_cross_modal_rep[uqidx.item()] = zs_rep[i].cpu().numpy()

    # save the representations to a file
    if save_zero_shot_rep == 'yes':
        if path_to_rep is not None and not os.path.exists(path_to_rep):
            logger.info(f"Saving zero-shot representations to {path_to_rep}")

            os.makedirs(os.path.dirname(path_to_rep), exist_ok=True)
            torch.save(saved_cross_modal_rep, path_to_rep)

    logger.info(f"Number of saved zero-shot representations: {len(saved_cross_modal_rep)}")
    logger.info(f"Shape of the dictionary used: {norm_dictionary_text_features.shape}")


    return saved_cross_modal_rep


def train(backbone, projector, ln_t, saved_cross_modal_rep, student_text_features, train_loader, real_test_loader, unlabelled_train_loader, args, device, wandb_logger, temperature=0.01):
    """Train the SpectralGCD model using a combination of supervised, unsupervised, and distillation losses.

    Args:
        backbone: Student CLIP model (typically ViT-B/16) to be fine-tuned.
        projector: MLP head that projects cross-modal representations to clustering and classification logits space.
            Returns (features, logits) where logits are used for classification/clustering.
        ln_t: Linear layer that projects raw cross-modal representations
            from dictionary_dim to a lower-dimensional space before the projector MLP.
        saved_cross_modal_rep: Dict mapping sample unique-index -> teacher cross-modal
            representation (np.ndarray). Precomputed from frozen teacher CLIP and cached.
        student_text_features: Text features [K, D] for the filtered concept dictionary,
            encoded by the student backbone. Used to compute student cross-modal representations.
        train_loader: DataLoader for training data with augmentations.
            Yields (images, labels, uq_idxs, mask_lab) where mask_lab indicates labeled samples.
        real_test_loader: DataLoader for the test set (no augmentation).
        unlabelled_train_loader: DataLoader for unlabeled training examples (no augmentation).
        args: Namespace with training hyperparameters including lr, lr_backbone, epochs,
            sup_weight, memax_weight, warmup_teacher_temp, teacher_temp, print_freq,
            test_frequency, load_checkpoint, start_epoch, fp16, and logger.
        device: Torch device for computation.
        wandb_logger: Weights & Biases logger for metrics tracking.
        temperature: Temperature scaling for cross-modal representations (default: 0.01).
            Must match the value used to compute saved_cross_modal_rep.

    Returns:
        None. Checkpoints are saved to args.model_path at intervals specified by test_frequency.
        Training and test metrics are logged to wandb_logger and args.logger.
    """
    ln_t_parameters = list(ln_t.parameters())
    projector_parameters = list(projector.parameters())
    backbone_parameters = list(backbone.parameters())
    
    if args.load_checkpoint is None:
        optimizer = SGD([
            {'params': backbone_parameters, 'lr': args.lr_backbone},
            {'params': projector_parameters, 'lr': args.lr},
            {'params': ln_t_parameters, 'lr': args.lr},
        ], momentum=args.momentum, weight_decay=args.weight_decay)
        args.logger.info(f"Optimizer: {optimizer}")
        args.logger.info(f"Using different learning rates for backbone and projector: {args.lr_backbone} for backbone, {args.lr} for projector and ln_t")

        exp_lr_scheduler = CustomCosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_mins=[args.lr_backbone * args.lr_eta_min, args.lr * args.lr_eta_min, args.lr * args.lr_eta_min]
        )

        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()
    else:
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        # Reconstruct optimizer with the same parameter groups
        optimizer = SGD([
            {'params': backbone_parameters, 'lr': args.lr_backbone},
            {'params': projector_parameters, 'lr': args.lr},
            {'params': ln_t_parameters, 'lr': args.lr},
        ], momentum=args.momentum, weight_decay=args.weight_decay)

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])

        args.logger.info("Optimizer state loaded from checkpoint.")

        exp_lr_scheduler = CustomCosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_mins=[args.lr_backbone * args.lr_eta_min, args.lr * args.lr_eta_min, args.lr * args.lr_eta_min]
        )

        if checkpoint.get('scheduler') is not None:
            args.logger.info("Loading scheduler state from checkpoint.")
            exp_lr_scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            # Fast-forward the scheduler to the correct epoch
            for _ in range(args.start_epoch):
                exp_lr_scheduler.step()

        if checkpoint.get('rng_state') is not None:
            args.logger.info("Loading RNG state from checkpoint.")
            torch.set_rng_state(checkpoint['rng_state'].cpu())
        if checkpoint.get('cuda_rng_state') is not None:
            args.logger.info("Loading CUDA RNG state from checkpoint.")
            cuda_rng_state = checkpoint['cuda_rng_state'].cpu()
            # Ensure it's a ByteTensor
            if not isinstance(cuda_rng_state, torch.ByteTensor):
                cuda_rng_state = cuda_rng_state.byte()
            torch.cuda.set_rng_state(cuda_rng_state)
        if checkpoint.get('numpy_rng_state') is not None:
            args.logger.info("Loading NumPy RNG state from checkpoint.")
            np.random.set_state(checkpoint['numpy_rng_state'])

        if checkpoint.get('random_rng_state') is not None:
            args.logger.info("Loading Python random state from checkpoint.")
            random.setstate(checkpoint['random_rng_state'])

        fp16_scaler = None
        if args.fp16:
            fp16_scaler = torch.cuda.amp.GradScaler()

        if checkpoint.get('fp16_scaler') is not None:
            fp16_scaler.load_state_dict(checkpoint['fp16_scaler'])
    



    cluster_criterion = DistillLoss(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    
    # Normalize text_features once before training loop, since they are fixed and do not require gradients
    norm_text_features = student_text_features / student_text_features.norm(dim=-1, keepdim=True)



    # TEST IF LOADING FROM CHECKPOINT
    if args.load_checkpoint is not None:
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc_mix, old_acc_mix, new_acc_mix = test(backbone, projector, ln_t, student_text_features, unlabelled_train_loader, epoch=args.start_epoch, save_name='Train ACC Unlabelled', args=args,device=device, temperature=temperature)


        args.logger.info('Train Accuracies(sum): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, new_acc_mix))

        if wandb_logger is not None:
            wandb_logger.log({"train_acc_all": all_acc_mix}, step=args.start_epoch)
            wandb_logger.log({"train_acc_old": old_acc_mix}, step=args.start_epoch)
            wandb_logger.log({"train_acc_new": new_acc_mix}, step=args.start_epoch)

        # TODO: Uncomment if you want to evaluate on a different test set during training (e.g., the real test set instead of the unlabelled training examples) 
        # args.logger.info('Testing on test data...')
        # test_all_acc_mix, test_old_acc_mix, test_new_acc_mix = test(backbone, projector, ln_t, student_text_features, real_test_loader, epoch=args.start_epoch, save_name='Test ACC', args=args,device=device, temperature=temperature)


        # args.logger.info('Test Accuracies(sum): All {:.4f} | Old {:.4f} | New {:.4f}'.format(test_all_acc_mix, test_old_acc_mix, test_new_acc_mix))

        # if wandb_logger is not None:
        #     wandb_logger.log({"test_acc_all": test_all_acc_mix}, step=args.start_epoch)
        #     wandb_logger.log({"test_acc_old": test_old_acc_mix}, step=args.start_epoch)
        #     wandb_logger.log({"test_acc_new": test_new_acc_mix}, step=args.start_epoch)



    # # inductive
    # best_test_acc_lab = 0
    # # transductive
    # best_train_acc_lab = 0
    # best_train_acc_ubl = 0 
    # best_train_acc_all = 0
    for epoch in range(args.start_epoch, args.epochs):
            
        loss_record = AverageMeter()

        visual_cls_loss_record = AverageMeter()
        visual_cluster_loss_record = AverageMeter()
        visual_sup_con_loss_record = AverageMeter()
        visual_contrastive_loss_record = AverageMeter()

        # KL distillation
        distill_loss_f_record = AverageMeter()
        distill_loss_b_record = AverageMeter()

        # Memax loss
        memax_loss_record = AverageMeter()

        backbone.train()
        projector.train()
        ln_t.train()



        
        
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True,device=device), mask_lab.cuda(non_blocking=True,device=device).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True,device=device)


            with torch.cuda.amp.autocast(fp16_scaler is not None):
                image_features = backbone.encode_image(images).float()
                norm_image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # Compute student cross-modal representation
                cross_modal_rep = norm_image_features @ norm_text_features.T
                cross_modal_rep = cross_modal_rep / temperature

                # Retrieve teacher cross-modal representations
                # Note that the student has two views of the same batch of images
                teacher_cross_modal_rep = torch.tensor(np.array([saved_cross_modal_rep[idx.item()] for idx in uq_idxs])).to(device)
                teacher_cross_modal_rep = torch.cat((teacher_cross_modal_rep, teacher_cross_modal_rep), dim=0)  # Repeat for two views
                
                

                student_log_prob = F.log_softmax(cross_modal_rep, dim=-1)
                teacher_log_prob = F.log_softmax(teacher_cross_modal_rep, dim=-1)

                # compute kl divergence loss
                distill_loss_f = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction='batchmean')
                distill_loss_b = F.kl_div(teacher_log_prob, student_log_prob, log_target=True, reduction='batchmean')
                forward_kl_loss = distill_loss_f
                reverse_kl_loss = distill_loss_b


                # Project cross-modal representation through linear layer
                proj_cross_modal_rep = ln_t(cross_modal_rep)

                
                student_proj, student_out = projector(proj_cross_modal_rep)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs))) # mean entropy reg for visual
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj,device=device)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels,device=device)
              
                loss_gcd = 0
                loss_gcd += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss_gcd += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
              
                
              
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                pstr += f'forward_KL_loss: {forward_kl_loss.item():.4f} '
                pstr += f'reverse_KL_loss: {reverse_kl_loss.item():.4f} '
                
                # Combine losses
                loss = loss_gcd + (forward_kl_loss + reverse_kl_loss)
  

            # Train loss recording
            loss_record.update(loss.item(), class_labels.size(0))
            
            visual_cls_loss_record.update(cls_loss.item(), class_labels.size(0))
            visual_cluster_loss_record.update(cluster_loss.item(), class_labels.size(0))
            visual_sup_con_loss_record.update(sup_con_loss.item(), class_labels.size(0))
            visual_contrastive_loss_record.update(contrastive_loss.item(), class_labels.size(0))
            # KL distillation
            distill_loss_f_record.update(forward_kl_loss.item(), class_labels.size(0))
            distill_loss_b_record.update(reverse_kl_loss.item(), class_labels.size(0))
            # Memax loss
            tmp_me_max_loss = args.memax_weight * me_max_loss
            memax_loss_record.update(tmp_me_max_loss.item(), class_labels.size(0))



            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
                


        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        
        if wandb_logger is not None:
            wandb_logger.log({"train_loss": loss_record.avg}, step=epoch)
            wandb_logger.log({"train_visual_cls_loss": visual_cls_loss_record.avg}, step=epoch)
            wandb_logger.log({"train_visual_cluster_loss": visual_cluster_loss_record.avg}, step=epoch)
            wandb_logger.log({"train_visual_sup_con_loss": visual_sup_con_loss_record.avg}, step=epoch)
            wandb_logger.log({"train_visual_contrastive_loss": visual_contrastive_loss_record.avg}, step=epoch)
            wandb_logger.log({"train_distill_loss_f": distill_loss_f_record.avg}, step=epoch)
            wandb_logger.log({"train_distill_loss_b": distill_loss_b_record.avg}, step=epoch)
            wandb_logger.log({"train_memax_loss": memax_loss_record.avg}, step=epoch)

            wandb_logger.log({"lr_backbone": optimizer.param_groups[0]['lr']}, step=epoch)
            wandb_logger.log({"lr_projector": optimizer.param_groups[1]['lr']}, step=epoch)
            wandb_logger.log({"lr_ln_t": optimizer.param_groups[2]['lr']}, step=epoch)
            wandb_logger.log({"train_epoch": epoch}, step=epoch)

        # Step schedule
        if exp_lr_scheduler is not None:
            exp_lr_scheduler.step()

        if args.test_frequency > 0 and (epoch % args.test_frequency == 0 or epoch == args.epochs - 1):


            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc_mix, old_acc_mix, new_acc_mix = test(backbone, projector, ln_t, student_text_features, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args,device=device, temperature=temperature)


            args.logger.info('Train Accuracies(sum): All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_mix, old_acc_mix, new_acc_mix))
            
            if wandb_logger is not None:
                wandb_logger.log({"train_acc_all": all_acc_mix}, step=epoch)
                wandb_logger.log({"train_acc_old": old_acc_mix}, step=epoch)
                wandb_logger.log({"train_acc_new": new_acc_mix}, step=epoch)

            # TODO: Uncomment if you want to evaluate on a different test set during training (e.g., the real test set instead of the unlabelled training examples) 
            # args.logger.info('Testing on test data...')
            # test_all_acc_mix, test_old_acc_mix, test_new_acc_mix = test(backbone, projector, ln_t, student_text_features, real_test_loader, epoch=epoch, save_name='Test ACC', args=args,device=device, temperature=temperature)


            # args.logger.info('Test Accuracies(sum): All {:.4f} | Old {:.4f} | New {:.4f}'.format(test_all_acc_mix, test_old_acc_mix, test_new_acc_mix))

            # if wandb_logger is not None:
            #     wandb_logger.log({"test_acc_all": test_all_acc_mix}, step=epoch)
            #     wandb_logger.log({"test_acc_old": test_old_acc_mix}, step=epoch)
            #     wandb_logger.log({"test_acc_new": test_new_acc_mix}, step=epoch)

            save_dict = {
                'backbone': backbone.state_dict(),
                'ln_t': ln_t.state_dict(),
                'proj': projector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'scheduler': exp_lr_scheduler.state_dict(),
                'rng_state': torch.get_rng_state(),  
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
                'fp16_scaler': fp16_scaler.state_dict() if fp16_scaler is not None else None,
            }

            torch.save(save_dict, args.model_path)
            args.logger.info("full model saved to {}.".format(args.model_path))
        
        



def test(backbone, projector, ln_t, student_text_features, test_loader, epoch, save_name, args, device, temperature):
    backbone.eval()
    projector.eval()
    ln_t.eval()

    norm_text_features = student_text_features / student_text_features.norm(dim=1, keepdim=True)
    
    targets = []
    preds_mix = []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True,device=device)
        with torch.no_grad():
            image_features = backbone.encode_image(images).float()
            norm_image_features = image_features / image_features.norm(dim=1, keepdim=True)
            
            cross_modal_rep = norm_image_features @ norm_text_features.T
            cross_modal_rep = cross_modal_rep / temperature

            proj_cross_modal_rep = ln_t(cross_modal_rep)
            
            ## -----combined_branch-----
            _, logits_mix = projector(proj_cross_modal_rep)


            preds_mix.append(logits_mix.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))


    preds_mix = np.concatenate(preds_mix)
    
    targets = np.concatenate(targets)
    all_acc_mix, old_acc_mix, new_acc_mix = log_accs_from_preds(y_true=targets, y_pred=preds_mix, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)

    return all_acc_mix, old_acc_mix, new_acc_mix

def set_random_seed(seed: int) -> None:
    import random,os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SpectralGCD training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # -- Dataset --
    parser.add_argument('--dataset_name', type=str, default='cub', help='Dataset to train on. Options: cifar10, cifar100, imagenet_100, cub, scars, aircraft.')
    parser.add_argument('--prop_train_labels', type=float, default=0.5, help='Proportion of labels used for the labelled (old) split.')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True, help='Use Semantic Shift Benchmark class splits.')

    # -- Dataloader --
    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size (before view duplication, effective batch = batch_size * n_views).')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of dataloader workers.')
    parser.add_argument('--n_views', default=2, type=int, help='Number of augmented views per image for contrastive learning.')
    parser.add_argument('--transform', type=str, default='imagenet', help='Augmentation strategy applied to training images.')

    # -- Model --
    parser.add_argument('--teacher_name', type=str, default='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', help='HuggingFace model ID for the frozen CLIP teacher that produces cross-modal representations.')
    parser.add_argument('--train_mode', type=str, default='last_vision_block', help='Which layers of the student CLIP backbone to fine-tune. Options: last_vision_block.')
    parser.add_argument('--grad_from_block', type=int, default=11, help='Transformer block index from which gradients flow (blocks before this are frozen).')
    parser.add_argument('--use_specialized_template', type=str, default='yes', help="Whether to use dataset-specific prompt templates for text encoding. Options: 'no', 'yes'.")

    # -- Optimizer (SGD) --
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for projector and concept projector (ln_t).')
    parser.add_argument('--lr_backbone', type=float, default=0.005, help='Learning rate for the student CLIP backbone (typically much smaller than lr).')
    parser.add_argument('--lr_eta_min', type=float, default=1e-3, help='Minimum lr ratio for cosine annealing (final_lr = lr * lr_eta_min).')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='L2 regularization weight.')

    # -- Training --
    parser.add_argument('--epochs', default=200, type=int, help='Total number of training epochs.')
    parser.add_argument('--sup_weight', type=float, default=0.35, help='Balance between supervised and unsupervised losses (0=fully unsup, 1=fully sup).')
    parser.add_argument('--memax_weight', type=float, default=2, help='Weight for mean-entropy maximization regularizer.')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable mixed-precision training. Legacy, not actively used.')

    # -- gcd head temperatures --
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial gcd head temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final gcd head temperature after warmup.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for gcd head temperature.')

    # -- Paths --
    parser.add_argument('--path_to_filtered_concepts', type=str, default='path_to_filtered_concepts/', help='CSV file with filtered concepts produced by spectral_filtering.py.')
    parser.add_argument('--path_to_saved_class_names', type=str, default='./dataset_class_names', help='Directory containing old/new class name CSVs.')
    parser.add_argument('--path_to_saved_cross_modal_representations', default='', type=str, help='Directory to cache teacher cross-modal representations. If empty, representations are computed but not saved. Delete cached files when changing the dictionary or teacher model.')
    parser.add_argument('--save_zero_shot_rep', type=str, default='yes', help="Whether to save computed cross-modal representations to disk. Options: 'no', 'yes'.")
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a training checkpoint to resume from (None = train from scratch).')

    # -- Experiment / logging --
    parser.add_argument('--exp_id', default='spectral_gcd', type=str, help='Experiment ID used as the log folder name.')
    parser.add_argument('--exp_root', type=str, default='./exps', help='Root directory to save experiment outputs.')
    parser.add_argument('--print_freq', default=10, type=int, help='Print training stats every N batches.')
    parser.add_argument('--test_frequency', default=5, type=int, help='Evaluate on test set every N epochs (0 = no testing during training).')
    parser.add_argument('--eval_funcs', nargs='+', default=['v2'], help='Evaluation metric(s) to use.')

    # -- Hardware / reproducibility --
    parser.add_argument('--seed', default=None, type=int, help='Random seed (None = non-deterministic with cudnn.benchmark=True).')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU device index.')

    # -- Weights & Biases --
    parser.add_argument('--use_wandb', action='store_true', default=False, help='If True, enables wandb logging.')
    parser.add_argument('--w_key_path', type=str, default=None, help='Path to file containing wandb API key. Expect a .txt file with the API key as a single line. Required if --use_wandb is set to True.')
    parser.add_argument('--project_name', default='custom_clip_gcd_concepts', type=str, help='Wandb project name.')
    parser.add_argument('--experiment_name', default='custom_clip_gcd_concepts', type=str, help='Wandb run name.')
    parser.add_argument('--group_name', default='example', type=str, help='Wandb group name.')
    
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()



    if args.seed is None:
        torch.backends.cudnn.benchmark = True
    else: set_random_seed(args.seed)
    
    device = torch.device(f"cuda:{args.cuda_dev}" if torch.cuda.is_available() else "cpu")

    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    
    init_experiment(args, runner_name=['SpectralGCD'], exp_id=args.exp_id)
    if args.use_wandb:
        l_dir = args.exp_root
        wandb_logger = set_wandb(args, log_dir=l_dir)
    else:
        wandb_logger = None
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


    args.logger.info(f"Running with args: {args}")
    
    
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets,train_examples_test = get_datasets(args.dataset_name,
                                                                                         train_transform,
                                                                                         test_transform,
                                                                                         args)
    

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    # TODO: Uncomment if you want to evaluate on a different test set during training (e.g., the real test set instead of the unlabelled training examples) 
    # real_test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
    #                                   batch_size=args.batch_size, shuffle=False)

    
    
    # load model
    model_clip, preprocess = load_clip_model("ViT-B/16", device=device)

    # Load zero-shot teacher model
    if args.teacher_name == "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K":
        args.logger.info(f"Loading teacher_clip with backbone: {args.teacher_name}")
        teacher_clip, _, _ = open_clip.create_model_and_transforms(args.teacher_name)
    else:
        raise ValueError(f"Unsupported teacher model: {args.teacher_name}")

    teacher_clip = teacher_clip.to(device)
    teacher_clip.eval()

    for m in teacher_clip.parameters():
        m.requires_grad = False

        
    model_clip = setup_trainable_parameters(model_clip, args.logger, args.train_mode, args.grad_from_block)
    
    
    args.logger.info("model_clip:")
    for name, m in model_clip.named_parameters():
        if m.requires_grad:
            args.logger.info(f"{name} requires_grad")


    args.logger.info("Teacher clip model:")
    for name, m in teacher_clip.named_parameters():
        if m.requires_grad:
            args.logger.info(f"{name} requires_grad")
    
    dictionary_text_features, tokenized_prompts = extract_dictionary_text_features(
        teacher_clip, args.logger, args.num_workers, args.batch_size, args.use_specialized_template,
        args.dataset_name, args.path_to_saved_class_names, device,
        path_to_dictionary=args.path_to_filtered_concepts
    )
    dictionary_text_features = dictionary_text_features.float().to(device)

            
    tmp_dataloader = DataLoader(train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, pin_memory=False, persistent_workers=True)


    args.dictionary_dim = dictionary_text_features.shape[0]
    args.logger.info(f"Number of concepts: {args.dictionary_dim}")


    # compute and save zero-shot teacher representations
    args.logger.info("Computing zero-shot teacher representations...")
    saved_cross_modal_rep = compute_zero_shot_teacher_rep(
        dictionary_text_features, tmp_dataloader, teacher_clip, args.logger,
        args.path_to_saved_cross_modal_representations, args.dataset_name, device,
        save_zero_shot_rep=args.save_zero_shot_rep, temperature=0.01
    )


    args.logger.info("Extracting student text features from the model_clip...")
    # Note that here we use the same tokenized prompts as for the teacher, since the two tokenizers are the same
    student_text_features = extract_student_dictionary_text_features(model_clip, tokenized_prompts, args.logger, device)
    student_text_features = student_text_features.float().to(device)
    
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projector = DINOHead(in_dim=768, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers).to(device)
    
    args.logger.info(f"Projector:")
    for name, m in projector.named_parameters():
        if m.requires_grad:
            args.logger.info(f"{name} requires_grad")

    ln_t = nn.Linear(in_features=args.dictionary_dim, out_features=768).to(device)

    args.logger.info(f"Linear text:")
    for name, m in ln_t.named_parameters():
        if m.requires_grad:
            args.logger.info(f"{name} requires_grad")

    
    # erase teacher_clip from memory
    del teacher_clip
    torch.cuda.empty_cache()
    args.logger.info("Starting training...")
    args.logger.info(f"Number of concepts: {dictionary_text_features.shape[0]}")
    args.logger.info(f"Number of student text features: {student_text_features.shape[0]}")


    if args.load_checkpoint is not None:
        args.logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        model_clip.load_state_dict(checkpoint['backbone'])
        projector.load_state_dict(checkpoint['proj'])
        ln_t.load_state_dict(checkpoint['ln_t'])
        args.start_epoch = checkpoint['epoch']
        args.logger.info(f"Loaded checkpoint from epoch {args.start_epoch}")
    else:
        args.start_epoch = 0

    train(model_clip, projector, ln_t, saved_cross_modal_rep, student_text_features, train_loader, None, test_loader_unlabelled, args, device, wandb_logger, temperature=0.01)
    # TODO: Uncomment if you want to evaluate on a different test set during training (e.g., the real test set instead of the unlabelled training examples) 
    # train(model_clip, projector, ln_t, saved_cross_modal_rep, student_text_features, train_loader, real_test_loader, test_loader_unlabelled, args, device, wandb_logger, temperature=0.01)
    