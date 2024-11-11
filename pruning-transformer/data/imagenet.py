import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, RandomResizedCrop, RandomHorizontalFlip
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torch.utils.data import Subset

# Refer to: https://github.com/pytorch/examples/blob/master/imagenet/main.py
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
normalize = transforms.Normalize(mean=MEAN, std=STD)

    
def get_dataset(data_path, train_folder="train", val_folder="val"):
    traindir = os.path.join(data_path, train_folder)
    valdir = os.path.join(data_path, val_folder)
    transforms_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transforms_val = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_set = datasets.ImageFolder(traindir, transforms_train)
    test_set = datasets.ImageFolder(valdir, transforms_val)

    return train_set, test_set

def get_sub_trainset(data_path, train_folder="train", val_folder="val", percentage=0.1, seed=42):
    import os
    traindir = os.path.join(data_path, train_folder)
    valdir = os.path.join(data_path, val_folder)
    
    transforms_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.ImageFolder(traindir, transform=transforms_train)

    file_path = "./data/imagenet_subset_indices.json"
    if os.path.exists(file_path) == False:
        indices_info = {}

        def get_subset(dataset):
            targets = torch.tensor(dataset.targets)
            class_indices = torch.unique(targets).tolist()  
            indices_per_class = [torch.where(targets == class_idx)[0].tolist() for class_idx in class_indices]
            subset_indices = []
            for class_idx, indices in zip(class_indices, indices_per_class):
                import random
                random.seed(seed)
                random.shuffle(indices) 
                selected_indices = indices[:int(len(indices) * percentage)]
                subset_indices.extend(selected_indices)  
                if class_idx not in indices_info:
                    indices_info[class_idx] = []
                indices_info[class_idx].extend(selected_indices)

            return Subset(dataset, subset_indices)

        train_subset = get_subset(train_set)

        import json
        with open(file_path, 'w') as f:
            json.dump(indices_info, f)

        return train_subset
    else:
        import json
        with open(file_path, 'r') as f:
            indices_info = json.load(f)

        # create a subset_indices list using indices_info
        subset_indices = []
        for indices in indices_info.values():
            subset_indices.extend(indices)
        train_subset = Subset(train_set, subset_indices)

        return train_subset

