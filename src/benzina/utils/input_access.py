import os

import torch


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def get_indices_by_names(dataset, filenames):
    """
    Retreive the indices of inputs by file names

    Args:
        dataset (benzina.torch.dataset.Dataset): dataset from which to fetch the indices.
        filenames (sequence): a sequence of file names
    """
    filenames_indices = {}
    filenames_lookup = set(filenames)

    with open(os.path.join(dataset.root, "data.filenames"), 'r') as names:
        for i, name in enumerate(names):
            # skip the end line following the filename
            name = name.rstrip()
            if name in filenames_lookup:
                filenames_indices[name] = i

    return [filenames_indices.get(filename, None) for filename in filenames]
