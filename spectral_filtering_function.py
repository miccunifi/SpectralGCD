"""
Simplified Spectral Filtering for CLIP-like Models

Single function that performs spectral filtering and returns ordered concepts.
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple


def spectral_filter(
    image_features: Union[np.ndarray, torch.Tensor],
    text_features: Union[np.ndarray, torch.Tensor],
    concept_names: List[str],
    normalize_features: bool = True,
    temperature: float = 0.01,
    beta_e: float = 0.95,
    beta_c: float = 0.99,
    apply_softmax: bool = True,
    use_approximation: bool = False,
    device: str = "cpu",
    batch_size: Optional[int] = None
) -> Tuple[Dict[str, float], int]:
    """
    Perform spectral filtering on image and text features.
    
    This function:
    1. Computes logits (cosine similarities) from features
    2. Performs PCA/eigendecomposition on the covariance matrix
    3. Ranks concepts by importance using spectral analysis
    4. Returns ordered dictionary of concepts with their importance scores
    
    Args:
        image_features: Image features [N, D] where N is number of images, D is feature dimension
        text_features: Text features [K, D] where K is number of concepts
        concept_names: List of K concept names corresponding to text features
        normalize_features: Whether to L2-normalize features before computing similarities (default: True)
        temperature: Temperature for scaling logits (default: 0.01)
        beta_e: Fraction of eigenvalue variance to retain (default: 0.95)
        beta_c: Fraction of concept importance to retain (default: 0.99)
        apply_softmax: Whether to apply softmax to logits before PCA (default: True)
        use_approximation: Whether to use approximate SVD for memory efficiency (default: False)
        device: Device for computation - 'cpu' or 'cuda:0' etc (default: 'cpu')
        batch_size: Batch size for computing logits. If None, compute all at once. 
                    Use smaller values (e.g., 100-500) for large datasets to avoid OOM (default: None)
        
    Returns:
        Tuple of:
            - ordered_concepts: List of concept names sorted by importance (descending)
            - n_concepts_retained: Number of top concepts that explain beta_c of importance

    Example:
        >>> image_feats = np.random.randn(1000, 512).astype(np.float32)
        >>> text_feats = np.random.randn(500, 512).astype(np.float32)
        >>> names = [f"concept_{i}" for i in range(500)]
        >>> ordered_concepts, k = spectral_filter(image_feats, text_feats, names)
        >>> print(f"Top {k} concepts:", ordered_concepts[:k])

        >>> # For large datasets, use batching:
        >>> ordered_concepts, k = spectral_filter(image_feats, text_feats, names, batch_size=100)
    """
    
    # Convert device to string if it's a torch.device object
    if isinstance(device, torch.device):
        device = str(device)

    # Step 1: Prepare features
    if isinstance(image_features, np.ndarray):
        image_features = torch.from_numpy(image_features).to(device)
    else:
        image_features = image_features.to(device)
        
    if isinstance(text_features, np.ndarray):
        text_features = torch.from_numpy(text_features).to(device)
    else:
        text_features = text_features.to(device)
    
    # Normalize if requested
    if normalize_features:
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Step 2: Compute logits (cosine similarities) in batches if needed
    n_images = image_features.shape[0]
    
    if batch_size is None or batch_size >= n_images:
        # Compute all at once (original behavior)
        logits = (image_features @ text_features.T) / temperature
        logits = logits.cpu().numpy()
    else:
        # Compute in batches to avoid OOM
        logits_list = []
        for i in range(0, n_images, batch_size):
            batch_end = min(i + batch_size, n_images)
            batch_image_features = image_features[i:batch_end]
            
            # Compute logits for this batch
            batch_logits = (batch_image_features @ text_features.T) / temperature
            logits_list.append(batch_logits.cpu().numpy())
            
        # Concatenate all batches
        logits = np.vstack(logits_list)
    
    # Step 3: Apply softmax if requested
    # Softmax operates row-wise so no batching needed (memory efficient already)
    if apply_softmax:
        logits = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    else:
        logits = logits.astype(np.float32)
    
    # Step 4: Center the data
    logits_centered = logits - logits.mean(axis=0, keepdims=True)
    del logits
    
    # Step 5: Compute eigendecomposition of the covariance matrix
    if use_approximation:
        # Use approximate SVD (memory efficient)
        logits_torch = torch.from_numpy(logits_centered).to(device)
        n_components = min(logits_torch.shape[0], logits_torch.shape[1])
        
        # torch.pca_lowrank returns U, S, V where X ≈ U @ diag(S) @ V.T
        # We need eigendecomposition of X.T @ X, which has eigenvectors V and eigenvalues S^2/(n-1)
        _, singular_values, eigenvectors = torch.pca_lowrank(logits_torch, q=n_components, center=False)
        eigenvalues = (singular_values ** 2) / (logits_torch.shape[0] - 1)
        
        eigenvalues = eigenvalues.cpu().numpy()
        eigenvectors = eigenvectors.cpu().numpy()
    else:
        # Use full eigendecomposition
        # Compute covariance matrix: Cov = (X.T @ X) / (n - 1)
        covariance_matrix = (logits_centered.T @ logits_centered) / (logits_centered.shape[0] - 1)
        del logits_centered
        
        if device.startswith("cuda"):
            covariance_matrix_torch = torch.tensor(covariance_matrix, device=device)
            # torch.linalg.eigh returns eigenvalues and eigenvectors of symmetric matrix
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix_torch)
            eigenvalues = eigenvalues.cpu().numpy()
            eigenvectors = eigenvectors.cpu().numpy()
        else:
            # np.linalg.eigh returns eigenvalues and eigenvectors of symmetric matrix
            eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    # (by default eigh returns them in ascending order)
    eigenvalue_sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[eigenvalue_sort_indices]
    eigenvectors_sorted = eigenvectors[:, eigenvalue_sort_indices]
    
    # Step 6: Determine number of principal components to retain
    eigenvalue_cumsum = np.cumsum(eigenvalues_sorted)
    total_variance = eigenvalue_cumsum[-1]
    variance_threshold = beta_e * total_variance
    n_components_retained = np.searchsorted(eigenvalue_cumsum, variance_threshold) + 1
    n_components_retained = min(n_components_retained, len(eigenvalues_sorted))
    
    # Step 7: Compute concept importance scores
    # Use top n_components_retained eigenvectors
    top_eigenvectors = eigenvectors_sorted[:, :n_components_retained]
    
    # Square each element of the eigenvectors
    squared_eigenvectors = top_eigenvectors ** 2
    
    # Weight by corresponding eigenvalues
    eigenvalue_diagonal = np.diag(eigenvalues_sorted[:n_components_retained])
    weighted_squared_eigenvectors = squared_eigenvectors @ eigenvalue_diagonal
    
    # Sum across components to get importance score for each concept
    concept_importance_scores = weighted_squared_eigenvectors.sum(axis=1)
    
    # Sort concepts by importance (descending order)
    concept_sort_indices = np.argsort(concept_importance_scores)[::-1]
    sorted_importance_scores = concept_importance_scores[concept_sort_indices]
    
    # Step 8: Determine number of top concepts to keep
    importance_cumsum = np.cumsum(sorted_importance_scores)
    total_importance = importance_cumsum[-1]
    importance_threshold = beta_c * total_importance
    n_concepts_retained = np.searchsorted(importance_cumsum, importance_threshold) + 1
    n_concepts_retained = min(n_concepts_retained, len(sorted_importance_scores))
    
    # Step 9: Create list of concept names sorted by importance
    ordered_concepts = [
        concept_names[concept_idx]
        for concept_idx in concept_sort_indices
    ]

    return ordered_concepts, n_concepts_retained


