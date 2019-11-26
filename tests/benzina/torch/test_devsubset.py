import os

import torch

import benzina.torch as bz


class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def test_devsubset_pytorch_loading():
    dataset_path = os.environ["DATASET_PATH"]

    with open("annex/devsubset_files", 'r') as devsubset_list:
        # skip the end line following the filename
        subset_targets_map = {line.rstrip()[5:]: int(line[:4]) for line in devsubset_list if line}

    subset_filenames = set(subset_targets_map.keys())
    
    with open(os.path.join(dataset_path, "data.filenames"), 'r') as filenames:
        subset_indices = []
        subset_targets = []
        for i, filename in enumerate(filenames):
            # skip the end line following the filename
            filename = filename.rstrip()
            if filename in subset_filenames:
                subset_indices.append(i)
                subset_targets.append(subset_targets_map[filename])
    
    dataset = bz.ImageNet(dataset_path)

    subset_sampler = SubsetSequentialSampler(subset_indices)

    subset_loader = bz.DataLoader(dataset,
                                  batch_size=100,
                                  sampler=subset_sampler,
                                  seed=0,
                                  shape=(256,256))
    
    for start_i, (images, targets) in zip(range(0, len(subset_sampler), 100), subset_loader):
        for i, (image, target) in enumerate(zip(images, targets)):
            assert image.size() == (3, 256, 256)
            assert image.sum().item() > 0
            assert target.item() == subset_targets[start_i + i]
