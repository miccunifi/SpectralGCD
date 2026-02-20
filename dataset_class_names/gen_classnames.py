import numpy as np
import os

def gen(dataset_name):
    if dataset_name == 'cub':
        
        from config import cub_root
        import re
        
        classes_file = os.path.join(cub_root, "CUB_200_2011/classes.txt")
        with open(classes_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        classnames = [re.split(r'\d+\s+\d+\.', line.strip())[1] for line in lines]
        assert len(classnames) == 200
    
    elif dataset_name == 'scars': 
        from config import car_root as car_dir
        import scipy.io as sio
        car_root = car_dir + "/cars_{}/"
        meta_default_path = car_dir + "/cars_{}.mat"
        
        meta_file = car_dir + "/devkit/cars_meta.mat"
        meta_data = sio.loadmat(meta_file)
        classnames = [name[0] for name in meta_data['class_names'][0]]
        assert len(classnames) == 196
        
    elif dataset_name == 'aircraft': 
        from config import aircraft_root
        
        ## only variant
        class_type = 'variant'  
        split = 'train'  

        classes_file = os.path.join(aircraft_root, f'images_{class_type}_{split}.txt')

        classes = set()
        with open(classes_file, 'r') as f:
            for line in f:
                class_name = ' '.join(line.split(' ')[1:]).strip()
                classes.add(class_name)
        classnames = sorted(classes)  
        assert len(classnames) == 100
        
    elif dataset_name == 'cifar10': 
        from torchvision.datasets import CIFAR10
        from config import cifar_10_root
        cifar10_dataset = CIFAR10(root=cifar_10_root, train=True, download=False)
        classnames = cifar10_dataset.classes
        assert len(classnames) == 10
        
    elif dataset_name == 'cifar100': 
        from torchvision.datasets import CIFAR100
        from config import cifar_100_root
        cifar100_dataset = CIFAR100(root=cifar_100_root, train=True, download=False)
        classnames = cifar100_dataset.classes
        assert len(classnames) == 100
        
    
    elif dataset_name == 'imagenet_100':
        np.random.seed(0)  
        imagenet_100_classes_idx = np.random.choice(range(1000), size=(100,), replace=False)
        imagenet_100_classes_idx = np.sort(imagenet_100_classes_idx)  
        with open(os.path.join(os.path.dirname(__file__), "imagenet_classes.txt"), "r") as f:
            imagenet_1k_classnames = [line.strip() for line in f.readlines()]
        classnames = [imagenet_1k_classnames[i] for i in imagenet_100_classes_idx]
        
        assert len(classnames) == 100
        


    else:

        raise NotImplementedError

    
    return np.array(classnames)

if __name__ == "__main__":
    print(gen("cub"))