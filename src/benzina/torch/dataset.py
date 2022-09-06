# -*- coding: utf-8 -*-
import io
from dataclasses import dataclass

import torch.utils.data
from bcachefs import Cursor

from benzina.utils.file import File, Track


class ClassificationDataset(torch.utils.data.Dataset):
    @dataclass
    class Item:
        input: memoryview = None
        target: int = -1

    def __init__(self, bch_cursor: Cursor):
        super(ClassificationDataset, self).__init__()
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

        return self.Item(sample, target)

    def __add__(self, other):
        raise NotImplementedError()

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


class ImageDataset(ClassificationDataset):
    @dataclass
    class Item(ClassificationDataset.Item):
        track: Track = None

    def __init__(self, bch_cursor: Cursor, input_label: str = "bzna_thumb"):
        ClassificationDataset.__init__(self, bch_cursor)
        self._input_label = input_label

    def __getitem__(self, index: int):
        item = ClassificationDataset.__getitem__(self, index)
        return self.Item(input=item.input, target=(item.target,),
                         track=Track(File(io.BytesIO(item.input)),
                                     self._input_label))


class ImageNet(ImageDataset):
    def __init__(self,
                 bch_cursor: Cursor,
                 split: ["train", "val", "test"] = "train",
                 input_label: str = "bzna_thumb"):
        ImageDataset.__init__(self, bch_cursor.cd(split), input_label)
