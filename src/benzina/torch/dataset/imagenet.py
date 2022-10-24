# -*- coding: utf-8 -*-
import bcachefs as bch
from .dataset import ImageDataset


class ImageNet(ImageDataset):
    def __init__(self,
                 bch_cursor: bch.Cursor,
                 split: ["train", "val", "test"] = "train",
                 input_label: str = "bzna_thumb"):
        ImageDataset.__init__(self, bch_cursor=bch_cursor.cd(split), input_label=input_label)
