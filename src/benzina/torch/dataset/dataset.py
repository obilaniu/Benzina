# -*- coding: utf-8 -*-
import io
from dataclasses import dataclass
from typing import Callable, Optional

import torch.utils.data
import bcachefs as bch

from benzina.utils.file import File, Track


class BenzinaDatasetMixin(torch.utils.data.Dataset):
    @dataclass
    class Item:
        input: memoryview = None
        track: Track = None
        aux: tuple = None

    def __init__(self, bch_cursor: bch.Cursor, input_label: str = "bzna_thumb"):
        torch.utils.data.Dataset.__init__(self)
        self._cursor = bch_cursor
        self._input_label = input_label

    @property
    def filename(self):
        return self._cursor.filename

    def __add__(self, other):
        del other
        raise NotImplementedError()

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        return BenzinaDatasetMixin.Item(input=item[0], aux=tuple(item[1:]),
            track=Track(File(io.BytesIO(item[0])), self._input_label))


class ClassificationDatasetMixin(torch.utils.data.Dataset):
    def __init__(self, bch_cursor: bch.Cursor, target_transform: Optional[Callable] = None):
        torch.utils.data.Dataset.__init__(self)
        self.target_transform = target_transform
        self._cursor = bch_cursor
        self._samples = self.find_samples()

    @property
    def filename(self):
        return self._cursor.filename

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        if self._cursor.closed:
            self._cursor.open()

        sample, target = self._samples[index]
        sample = self._cursor.read(sample.inode)

        if self.target_transform:
            target = self.target_transform(target)

        return sample, target

    def find_samples(self):
        classes = [de for de in self._cursor.scandir()
                   if de.is_dir and de.name != "lost+found"]
        class_to_idx = {cls.name: i for i, cls in enumerate(classes)}

        instances = []
        available_classes = set()
        for target_class in classes:
            class_idx = class_to_idx[target_class.name]
            for _, _, files in \
                sorted(self._cursor.walk(target_class.name)):
                for f in sorted(files, key=lambda _f: _f.name):
                    item = f, class_idx
                    instances.append(item)
                    if target_class.name not in available_classes:
                        available_classes.add(target_class.name)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            raise FileNotFoundError(msg)

        return instances


class ImageDataset(BenzinaDatasetMixin, ClassificationDatasetMixin):
    def __init__(self, bch_cursor: bch.Cursor, target_transform: Optional[Callable] = None, input_label: str = "bzna_thumb"):
        BenzinaDatasetMixin.__init__(self, bch_cursor=bch_cursor, input_label=input_label)
        ClassificationDatasetMixin.__init__(self, bch_cursor=bch_cursor, target_transform=target_transform)
