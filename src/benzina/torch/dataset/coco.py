# -*- coding: utf-8 -*-
import json, time
from typing import Callable, List, Optional
from warnings import warn

import bcachefs as bch
import torchvision
from pycocotools.coco import COCO

from .dataset import BenzinaDatasetMixin


class CocoDetection(BenzinaDatasetMixin, torchvision.datasets.CocoDetection):
    def __init__(
        self,
        bch_cursor: bch.Cursor,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        input_label: str = "bzna_thumb",
        *args, **kwargs
    ) -> None:
        if transform or transforms:
            warn("Benzina doesn't support torchvision transforms on images yet. "
                 "Removing `transform` and image transform from `transforms` args")
            transform = None
            if transforms:
                transforms.transform = None
        BenzinaDatasetMixin.__init__(self, bch_cursor=bch_cursor,
            input_label=input_label)
        torchvision.datasets.CocoDetection.__init__(self, *args,
            root=bch_cursor.filename, annFile=None, transform=transform,
            target_transform=target_transform, transforms=transforms, **kwargs)

        self._cursor = bch_cursor
        print('loading annotations into memory...')
        tic = time.time()
        with bch.Bcachefs(bch_cursor.filename) as bchfs:
            with bchfs.open(annFile, 'r') as f:
                dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        self.coco.dataset = dataset
        self.coco.createIndex()
        self.ids = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, id: int):
        if self._cursor.closed:
            self._cursor.open()
        path = self.coco.loadImgs(id)[0]["file_name"]
        return self._cursor.read(path + ".mp4")


class CocoCaptions(CocoDetection):
    def _load_target(self, id: int) -> List[str]:
        return [ann["caption"] for ann in super()._load_target(id)]