import numpy as np
from torch.utils.data import Dataset

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class ClassNamesDataset(Dataset):

    """
    Takes a list of class_names and create a dataset from it.
    """

    def __init__(self, class_names, target_transform_dict=None):

        self.class_names = class_names

        if target_transform_dict is not None:
            self.target_transform_dict = target_transform_dict
            assert len(class_names) == len(target_transform_dict)
            self.transformed_class_names = [None] * len(class_names)
            for old_index, new_index in target_transform_dict.items():
                self.transformed_class_names[new_index] = class_names[old_index]
        else:
            self.transformed_class_names = class_names


    def __getitem__(self, item):
        return self.transformed_class_names[item]

    def __len__(self):
        return len(self.class_names)