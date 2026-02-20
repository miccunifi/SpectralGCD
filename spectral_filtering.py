import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from utils.general_utils import init_experiment

import open_clip

import csv

from utils.general_utils import extract_dictionary_text_features




def compute_zero_shot_teacher_rep(dictionary_text_features, dataloader, model_clip, device, logger, temperature=0.01):
    """Compute zero-shot cross-modal representations by measuring cosine similarity
    between CLIP image features and dictionary text features for each sample."""
    model_clip.eval()

    cross_modal_rep = {}
    norm_dictionary_text_features = dictionary_text_features / dictionary_text_features.norm(dim=-1, keepdim=True)

    for idx_batch, batch in enumerate(tqdm(dataloader)):
        images, _, uqidxs, _ = batch
        images = images.cuda(non_blocking=True, device=device)

        with torch.no_grad():
            feats = model_clip.encode_image(images).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)

            # NOTE: 0.01 temperature scaling is applied, both to student and teacher since they use the same scale
            zs_cross_modal_rep = (feats @ norm_dictionary_text_features.T) / temperature

            for i, uqidx in enumerate(uqidxs):
                cross_modal_rep[uqidx.item()] = zs_cross_modal_rep[i].cpu().numpy()

    logger.info(f"Number of saved zero-shot cross-modal representations: {len(cross_modal_rep)}")
    logger.info(f"Shape of the dictionary used: {norm_dictionary_text_features.shape}")

    return cross_modal_rep


def variance_thresholding(S, variance_threshold=0.99, use_squared=True):
     # Compute explained variance ratio
    if use_squared:
        explained_variance = S**2
    else:
        explained_variance = S
    cumulative_variance = np.cumsum(explained_variance, axis=0)
    total_variance = cumulative_variance[-1]
    variance_ratio = cumulative_variance / total_variance
    # Find minimum number of singular values to preserve desired variance
    k = np.searchsorted(variance_ratio, variance_threshold).item() + 1
    return k
    

def spectral_filtering(pre_computed_cross_modal_rep, all_class_names, args, thresholding_concepts=0.99,
                       folder_name="spectral_filtering_results", thresholding_eig=0.99,
                       use_torch_impl=True, device="cuda:0", use_approximation=False):
    """Perform spectral filtering on pre-computed cross-modal representations to
    identify and rank the most important concepts for the dataset.

    The pipeline works as follows:
        1. Stack per-sample cross-modal representations and apply softmax normalization.
        2. Center the data (zero-mean per concept dimension).
        3. Compute the eigendecomposition of the covariance matrix (full or approximate).
        4. Retain the top principal components that explain >= beta_e of the total variance.
        5. Score each concept by its weighted contribution across retained components.
        6. Keep the top concepts that cumulatively explain >= beta_c of the total importance.
        7. Save the filtered concept list to a CSV file.

    Args:
        pre_computed_cross_modal_rep: Dict mapping sample unique-index -> cross-modal representation vector (np.ndarray).
            Produced by compute_zero_shot_teacher_rep.
        all_class_names: List of concept/class name strings, one per column of the cross-modal representations.
            len(all_class_names) must equal the dimensionality of each cross-modal vector.
        args: Namespace with at least `args.logger` (logging) and `args.dataset_name` (str).
        thresholding_concepts: Fraction of cumulative concept importance to retain (beta_c).
            Higher values keep more concepts. (default: 0.99)
        folder_name: Directory path where the output CSV will be saved. Created if missing.
            (default: "spectral_filtering_results")
        thresholding_eig: Fraction of cumulative eigenvalue variance to retain (beta_e).
            Controls how many principal components are kept. (default: 0.99)
        use_torch_impl: If True, move the covariance matrix to GPU and use torch.linalg.eigh
            instead of np.linalg.eigh. Faster for large matrices with GPU available.
            Only used when use_approximation is False. (default: False)
        device: Torch device string for GPU-accelerated computation, e.g. "cuda:0".
            Used when use_torch_impl=True or use_approximation=True. (default: "cuda:0")
        use_approximation: If True, use torch.pca_lowrank for an approximate SVD instead of
            computing the full covariance matrix. More memory-efficient for very large
            datasets. (default: False)
    """
    path_to_filtered_concepts = folder_name
    if not os.path.exists(path_to_filtered_concepts):
        os.makedirs(path_to_filtered_concepts, exist_ok=True)

    args.logger.info("Computing Spectral Filtering on cm representations...")

    # Stack per-sample representations into a single matrix [N, K]
    cross_modal_rep = np.vstack(list(pre_computed_cross_modal_rep.values()))

    # Apply softmax normalization across concepts (row-wise)
    cross_modal_rep = nn.Softmax(dim=1)(torch.tensor(cross_modal_rep)).numpy()

    # Center the data: zero-mean per concept dimension
    cm_rep_m = cross_modal_rep - cross_modal_rep.mean(axis=0, keepdims=True)
    del cross_modal_rep

    # Eigendecomposition of the covariance matrix
    if use_approximation:
        args.logger.info("Using low-rank approximation")
        cm_rep_m = torch.tensor(cm_rep_m, device=device)
        cm_rep_m = cm_rep_m.to(device)
        q = min(cm_rep_m.shape[0], cm_rep_m.shape[1])

        # torch.pca_lowrank: X ≈ U @ diag(S) @ V.T
        # Covariance eigenvectors = V, eigenvalues = S^2 / (n-1)
        _, singular_values, eigenvectors = torch.pca_lowrank(cm_rep_m, q=q, center=False)
        eigenvalues = (singular_values ** 2) / (cm_rep_m.shape[0] - 1)

        eigenvalues = eigenvalues.cpu().numpy()
        eigenvectors = eigenvectors.cpu().numpy()
    else:
        args.logger.info("Computing covariance matrix...")
        cov_matrix = (cm_rep_m.T @ cm_rep_m) / (cm_rep_m.shape[0] - 1)
        del cm_rep_m

        if use_torch_impl:
            args.logger.info(f"Using torch implementation on device: {device}")
            cov_matrix = torch.tensor(cov_matrix, device=device)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
            eigenvalues = eigenvalues.cpu().numpy()
            eigenvectors = eigenvectors.cpu().numpy()
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues/eigenvectors in descending order (eigh returns ascending)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    # Retain top-k principal components that explain >= thresholding_eig of variance
    k = variance_thresholding(eigenvalues_sorted, variance_threshold=thresholding_eig, use_squared=False)
    args.logger.info(f"K after thresholding: {k}")

    # Score each concept: sum of squared eigenvector loadings weighted by eigenvalues
    top_eigenvectors = eigenvectors_sorted[:, :k]
    eigenvalue_diagonal = np.diag(eigenvalues_sorted[:k])
    squared_eigenvectors = top_eigenvectors ** 2
    weighted_squared_eigenvectors = squared_eigenvectors @ eigenvalue_diagonal
    concept_importance_scores = weighted_squared_eigenvectors.sum(axis=1)

    # Sort concepts by importance (descending)
    concept_sort_indices = np.argsort(concept_importance_scores)[::-1]
    sorted_importance_scores = concept_importance_scores[concept_sort_indices]

    # Retain top concepts that explain >= thresholding_concepts of total importance
    k_concepts = variance_thresholding(sorted_importance_scores, variance_threshold=thresholding_concepts, use_squared=False)
    args.logger.info(f"Number of concepts after thresholding: {k_concepts}")

    # Save filtered concepts to CSV
    top_k_concepts_list = [all_class_names[i.item()] for i in concept_sort_indices[:k_concepts]]
    csv_file_path = os.path.join(path_to_filtered_concepts, f"{args.dataset_name}_concepts.csv")
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ClassName'])
        for concept in top_k_concepts_list:
            writer.writerow([concept])

    args.logger.info(f"Concepts saved to {csv_file_path}")



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

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--exp_root', type=str, default='./exps', help='Root directory to save experiment outputs.')
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)

    parser.add_argument('--exp_id', default='spectral_filtering', type=str)
    
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')

    parser.add_argument('--teacher_name', type=str, default='hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', help='How to train the model, options: ViT-H-14-quickgelu, hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    parser.add_argument('--use_specialized_template', type=str, default='no', help="'no', 'yes'")

    parser.add_argument('--thresholding_eig', type=float, default=0.99)
    parser.add_argument('--thresholding_concepts', type=float, default=0.99)

    parser.add_argument('--use_torch_impl', action='store_true', default=False)
    parser.add_argument('--use_lowrank_impl', action='store_true', default=False)

    # For saving results of Spectral Filtering
    parser.add_argument('--path_to_filtered_concepts', type=str, default='./path_to_filtered_concepts', help="Directory where the filtered concepts CSV will be saved.")
    # Path to the dictionary used for Spectral Filtering (expects a CSV with a "ClassName" column)
    parser.add_argument('--path_to_dictionary', default='', type=str, help="Path to the dictionary of concepts to filter.")
    # For old class names
    parser.add_argument('--path_to_saved_class_names', type=str, default='./dataset_class_names')
    
    
    
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
    
    init_experiment(args, runner_name=['SpectralFiltering'], exp_id=args.exp_id)
    
    args.interpolation = 3
    args.crop_pct = 0.875

    args.image_size = 224
    args.feat_dim = 512
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes


    args.logger.info(f"Running with args: {args}")
    
    
    # --------------------
    # TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    # --------------------
    # DATASETS
    # --------------------
    _, _, _, _,train_examples_test = get_datasets(args.dataset_name,
                                                    train_transform,
                                                    test_transform,
                                                    args)
    
    
    # Load zero-shot teacher model
    if args.teacher_name == "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K":
        teacher_backbone = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        args.logger.info(f"Loading teacher_clip with backbone: {teacher_backbone}")
        teacher_clip, _, _ = open_clip.create_model_and_transforms(teacher_backbone)
    else:
        raise ValueError(f"Invalid teacher choice: {args.teacher_name}. Choose from 'hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K'.")
    
    teacher_clip = teacher_clip.to(device)
    teacher_clip.eval()

    for m in teacher_clip.parameters():
        m.requires_grad = False

    args.logger.info("zs model_clip:")
    for name, m in teacher_clip.named_parameters():
        if m.requires_grad ==True:
            args.logger.info(f"{name} requires_grad")
    

    # NOTE: changed to path_to_dictionary to be passed as an argument
    dictionary_text_features, tokenized_prompts, all_class_names = extract_dictionary_text_features(
        teacher_clip, args.logger, args.num_workers, args.batch_size, args.use_specialized_template,
        args.dataset_name, args.path_to_saved_class_names, device,
        path_to_dictionary=args.path_to_dictionary, return_class_names=True
    )
    dictionary_text_features = dictionary_text_features.float().to(device)
            

    dataloader = DataLoader(train_examples_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                                drop_last=False, pin_memory=False)

    args.dictionary_dim = dictionary_text_features.shape[0]
    args.logger.info(f"Number of concepts: {args.dictionary_dim}")


    # Compute and save zero-shot teacher representations

    args.logger.info("Computing zero-shot cross-modal representations...")
    cross_modal_rep = compute_zero_shot_teacher_rep(dictionary_text_features, dataloader, teacher_clip, device, args.logger, temperature=0.01)


    assert cross_modal_rep[0].shape[0] == args.dictionary_dim, f"cross_modal_rep shape {cross_modal_rep[0].shape[0]} does not match dictionary dim {args.dictionary_dim}"
    assert cross_modal_rep[0].shape[0] == len(all_class_names), f"cross_modal_rep shape {cross_modal_rep[0].shape[0]} does not match number of class names {len(all_class_names)}"

    # remove both models from GPU memory
    del teacher_clip
    torch.cuda.empty_cache()


    args.logger.info(f"Thresholding eigenvalues and eigenvectors with threshold: {args.thresholding_eig}")
    args.logger.info(f"Thresholding concepts with threshold: {args.thresholding_concepts}")

    spectral_filtering(
        cross_modal_rep, all_class_names, args, thresholding_concepts=args.thresholding_concepts, 
        folder_name=args.path_to_filtered_concepts, thresholding_eig=args.thresholding_eig,
        use_torch_impl=args.use_torch_impl, device=device, use_approximation=args.use_lowrank_impl
    )