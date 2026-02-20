import os
import torch
import inspect

from datetime import datetime
from loguru import logger
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip import tokenize
import pandas as pd
from data.data_utils import ClassNamesDataset
import math


# Set up WandB for logging metrics
def set_wandb(args, log_dir=None):
    
    # Handle WandB API key from a file
    if hasattr(args, 'w_key_path') and args.w_key_path is not None and os.path.exists(args.w_key_path):
        try:
            with open(args.w_key_path, "r") as f:
                wandb_key = f.read().strip()
            wandb.login(key=wandb_key)
        except Exception as e:
            print(f"Error reading WandB key from {args.w_key_path}: {e}")
            return None
    else:
        print("Warning: WandB key path is not provided or does not exist. Ensure WandB is already logged in.")
    
    # Default log directory
    if log_dir is None:
        log_dir = '.wandb'
    elif not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} does not exist.")

    
    
    # Initialize WandB
    wandb.init(
        project=args.project_name,
        name=args.experiment_name,
        group=args.group_name,
        config={k: v for k, v in vars(args).items() if not k.startswith('_')},
        dir=log_dir,
    )

    return wandb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    root_dir = os.path.join(args.exp_root, *runner_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    log_dir = os.path.join(root_dir, 'log', f'{exp_id}')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')

    hparam_dict = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v

    print(runner_name)
    print(args)

    return args


    


def extract_dictionary_text_features(model, logger, num_workers, batch_size, use_specialized_template,
                                     dataset_name, path_to_saved_class_names, device, path_to_dictionary,
                                     return_class_names=False):
    """
    Extract text features for all concepts in the specified concepts dictionary.
    """
    model.eval()
    logger.info(f"Loading concepts text features from {path_to_dictionary}")
    all_class_names = extract_class_names(logger, path_to_saved_class_names, dataset_name, path_to_dictionary)

    dataset_class_names = ClassNamesDataset(all_class_names)


    class_names_loader = DataLoader(dataset_class_names, num_workers=num_workers, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True)

    all_features = []
    tokenized_prompts = []

    for idx, batch_cls_names in enumerate(tqdm(class_names_loader)):
        # Extract features
        with torch.no_grad():
            batch_prompts = []

            if use_specialized_template == 'yes':
                if dataset_name == 'scars':
                    batch_prompts = [f"a photo of a {cls_name} ." for cls_name in batch_cls_names]
                elif dataset_name == 'cub':
                    batch_prompts = [f"a photo of a {cls_name}, a type of bird." for cls_name in batch_cls_names]
                elif dataset_name == 'aircraft':
                    batch_prompts = [f"a photo of a {cls_name}, a type of aircraft." for cls_name in batch_cls_names]
                elif dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'imagenet_100':
                    # logger.info(f'{dataset_name} does not have a specialized template, using the standard one')
                    batch_prompts = [f"a photo of a {cls_name} ." for cls_name in batch_cls_names]
                else:
                    # Fallback to standard template if dataset is unrecognized
                    batch_prompts = [f"a photo of a {cls_name} ." for cls_name in batch_cls_names]
            elif use_specialized_template == 'no':
                batch_prompts = [f"a photo of a {cls_name} ." for cls_name in batch_cls_names]
            # NOTE: openai clip tokenizer, is the same as the one used for hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K
            batch_tokenized_prompts = tokenize(batch_prompts).to(device)
            tokenized_prompts.append(batch_tokenized_prompts)

            tmp_text_features = model.encode_text(batch_tokenized_prompts).float()
            all_features.append(tmp_text_features)

    # Concatenate all batches into a single tensor
    all_concepts_text_feat = torch.cat(all_features, dim=0)

    assert len(all_concepts_text_feat) == len(all_class_names), f"Mismatch in number of text features and class names: {len(all_concepts_text_feat)} vs {len(all_class_names)}"
    logger.info(f"Number of concepts: {len(all_concepts_text_feat)}")

    text_features = all_concepts_text_feat.to(device)

    tokenized_prompts = torch.cat(tokenized_prompts, dim=0).to(device)
    if return_class_names:
        return text_features, tokenized_prompts, all_class_names
    else:
        return text_features, tokenized_prompts


def extract_class_names(logger, path_to_saved_class_names, dataset_name, path_to_data):
    dict_concepts_names = pd.read_csv(path_to_data)
    dict_concepts_names = [str(name) for name in dict_concepts_names['ClassName'].values.tolist()]

    # Initialize the list that will hold all unique, ordered class names
    all_class_names = []
    seen_names = set()

    # Add initial concepts, preserving their order
    for name in dict_concepts_names:
        if name.lower() not in seen_names:
            all_class_names.append(name)
            seen_names.add(name.lower())

    # Add old class names, preserving their order and avoiding duplicates
    logger.info(f"Using old class names.")
    old_concepts_names = pd.read_csv(f"{path_to_saved_class_names}/{dataset_name}/old_class_names.csv")
    old_concepts_names = old_concepts_names['ClassName'].values.tolist()
    counter = 0
    added_class_names = []
    for name in old_concepts_names:
        name = name.replace('_', ' ')  # Replace underscores with spaces
        if name.lower() not in seen_names:
            all_class_names.append(name)
            seen_names.add(name.lower())
            added_class_names.append(name)
            counter += 1
    return all_class_names




def extract_student_dictionary_text_features(model, tokenized_prompts, logger, device):
    model.eval()
    logger.info(f"Number of concepts: {len(tokenized_prompts)}")

    all_text_features = []
    num_prompts = tokenized_prompts.shape[0]
    tmp_batch_size = 512
    num_batches = math.ceil(num_prompts / tmp_batch_size)


    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * tmp_batch_size
            end_idx = min((i + 1) * tmp_batch_size, num_prompts)
            batch_prompts = tokenized_prompts[start_idx:end_idx]

            tmp_text_features_batch = model.encode_text(batch_prompts).float()
            all_text_features.append(tmp_text_features_batch)

            logger.info(f"Processed batch {i+1}/{num_batches} (prompts {start_idx}-{end_idx-1})")

    # Concatenate all batch features into a single tensor
    text_features = torch.cat(all_text_features, dim=0)

    logger.info(f"Number of concepts: {text_features.shape[0]}")

    text_features = text_features.to(device)

    return text_features



class CustomCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_mins, last_epoch=-1):
        self.T_max = T_max
        self.eta_mins = eta_mins
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            eta_min + (base_lr - eta_min) * 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
        ]
    

