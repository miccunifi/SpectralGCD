import argparse
import os
import numpy as np
import torch

from data.get_datasets import get_class_splits
import pandas as pd

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

    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herbarium_19')
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--path_to_saved_class_names', type=str, default='./dataset_class_names/')
    
    
    
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    
    # get classnames
    from dataset_class_names import gen_classnames
    class_names = gen_classnames.gen(args.dataset_name)

    # Remove _ from class names and replace with space
    class_names = np.array([name.replace('_', ' ') for name in class_names])

    print(f"Class names generated for {args.dataset_name}, total {len(class_names)} classes.")
    print(f"Class names: {class_names}")

    args.base_names = class_names[args.train_classes]

    print(f"Base class names: {args.base_names}")
    print(f"Number of base classes: {len(args.base_names)}")

    # save the old and new class names to two different csv files

    path_to_filtered_concepts = os.path.join(args.path_to_saved_class_names, args.dataset_name)
    if not os.path.exists(path_to_filtered_concepts):
        os.makedirs(path_to_filtered_concepts, exist_ok=True)
        print(f"Created directory: {path_to_filtered_concepts}")

    df_old_class_names = pd.DataFrame(args.base_names, columns=["ClassName"])

    df_old_class_names.to_csv(os.path.join(path_to_filtered_concepts, "old_class_names.csv"), index=False)

    print(f"Transformed class names saved to {os.path.join(path_to_filtered_concepts, 'old_class_names.csv')}")
    
    args.new_class_names = class_names[args.unlabeled_classes]
    print(f"New class names: {args.new_class_names}")
    df_new_class_names = pd.DataFrame(args.new_class_names, columns=["ClassName"])
    df_new_class_names.to_csv(os.path.join(path_to_filtered_concepts, "new_class_names.csv"), index=False)
    