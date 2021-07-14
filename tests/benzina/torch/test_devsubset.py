import os

import benzina.torch as bz
from benzina.utils.input_access import SubsetSequentialSampler, get_indices_by_names


def test_devsubset_pytorch_loading():
    dataset_path = os.environ["DATASET_PATH"]

    with open("annex/devsubset_files", 'r') as devsubset_list:
        subset_filenames = []
        subset_targets = []
        # skip the end line following the filename
        for line in devsubset_list:
            if not line:
                continue
            subset_filenames.append(line.rstrip()[5:])
            subset_targets.append(int(line[:4]))

    dataset = bz.ImageNet(dataset_path)
    subset_indices = get_indices_by_names(dataset, subset_filenames)

    subset_sampler = SubsetSequentialSampler(subset_indices)

    subset_loader = bz.DataLoader(dataset,
                                  batch_size=100,
                                  sampler=subset_sampler,
                                  seed=0,
                                  shape=(256, 256))

    for start_i, (images, targets) in zip(range(0, len(subset_sampler), 100), subset_loader):
        for i, (image, target) in enumerate(zip(images, targets)):
            assert image.size() == (3, 256, 256)
            assert image.sum().item() > 0
            assert target.item() == subset_targets[start_i + i]
