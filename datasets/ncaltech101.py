import os
import hdf5plugin
import h5py
import numpy as np
import torch
import lightning as L

from enum import Enum
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from utils.data import GraphData
from utils.convert_bbox import convert_to_training_format
from datasets.graph_gen.matrix_neighbour import GraphGenerator
from datasets.augmentations.augmentation import (
    Compose, RandomHFlip, RandomCrop, RandomZoom, RandomTranslate, Crop
)

class SliceMethod(Enum):
    FIRST_BY_TIME = "first_by_time"
    LAST_BY_TIME = "last_by_time"
    MID_BY_TIME = "mid_by_time"
    FIRST_BY_COUNT = "first_by_count"
    LAST_BY_COUNT = "last_by_count"
    MID_BY_COUNT = "mid_by_count"


class NCaltech101Dataset(Dataset):
    """
    Dataset for NCaltech101 that loads binary event files and annotation files,
    generates graphs using GraphGenerator, and returns per-sample dicts.

    Expected directory layout:
        data_dir/Caltech101/{class_name}/image_XXXX.bin
        data_dir/Caltech101_annotations/{class_name}/annotation_XXXX.bin
    """

    def __init__(self, samples, class_to_idx, cfg, augment=False):
        """
        Args:
            samples: list of (event_path, annotation_path, class_idx) tuples
            class_to_idx: dict mapping class name -> int index
            cfg: full nested config dict
            augment: whether to apply data augmentation
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.augment = augment

        data_cfg  = cfg.get("data",  {})
        norm_cfg  = cfg.get("norm",  {})
        graph_cfg = cfg.get("graph", {})
        aug_cfg   = cfg.get("augmentation", {})

        self.width  = data_cfg.get("sensor_width",  240)
        self.height = data_cfg.get("sensor_height", 180)

        self.norm_w = norm_cfg.get("norm_w", 240)
        self.norm_h = norm_cfg.get("norm_h", 180)
        self.norm_t = norm_cfg.get("norm_t", 1000)

        self.num_events   = data_cfg.get("num_events",   50000)
        self.sample_len   = data_cfg.get("sample_len",   100000)
        self.slice_method = SliceMethod(data_cfg.get("slice_method", "mid_by_time"))

        self.radius_x = graph_cfg.get("radius_x", 3)
        self.radius_y = graph_cfg.get("radius_y", 3)
        self.radius_t = graph_cfg.get("radius_t", 5)

        self.generator = GraphGenerator(width=self.width, height=self.height)

        if augment:
            self.transform = Compose([
                RandomHFlip(p=aug_cfg.get("hflip_p", 0.5), width=self.width),
                RandomCrop(size=aug_cfg.get("crop_size", [0.75, 0.75]),
                           p=aug_cfg.get("crop_p", 0.2),
                           width=self.width, height=self.height),
                RandomZoom(zoom=tuple(aug_cfg.get("zoom_range", [1.0, 1.5])),
                           subsample=aug_cfg.get("zoom_subsample", True),
                           width=self.width, height=self.height),
                RandomTranslate(size=aug_cfg.get("translate_size", [0.1, 0.1]),
                                width=self.width, height=self.height),
                Crop([0, 0], [1, 1], width=self.width, height=self.height),
            ])
        else:
            self.transform = Compose([
                Crop([0, 0], [1, 1], width=self.width, height=self.height),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ev_path, ann_path, class_idx = self.samples[idx]

        events = NCaltech101.load_events(ev_path, self.num_events)
        bbox = NCaltech101.load_annotations(ann_path)

        # Limit number of events
        # events = self.slice_events(events)

        # Augment in pixel space before normalization
        bbox = torch.tensor(np.append(bbox, class_idx), dtype=torch.float32).unsqueeze(0)
        events, bbox = self.transform(events, bbox)

        if events.shape[0] == 0:
            return self.__getitem__((idx + 1) % len(self))

        events = self.normalize_events(events)

        frame = self._make_event_frame(events)

        # Generate graph
        x, pos, edge_index = self.generator.generate_edges(
            events,
            radius_x=self.radius_x,
            radius_y=self.radius_y,
            radius_t=self.radius_t,
        )

        self.generator.clear()
        return GraphData(x=x, pos=pos, edge_index=edge_index, bboxes=bbox, frame=frame)
    
    def _make_event_frame(self, events) -> np.ndarray:
        """Render normalized events into an RGB image [H, W, 3] uint8."""
        h, w = int(self.norm_h), int(self.norm_w)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        xs = events[:, 0].numpy().astype(int).clip(0, w - 1)
        ys = events[:, 1].numpy().astype(int).clip(0, h - 1)
        ps = events[:, 3].numpy()
        neg = ps < 0
        frame[ys[neg], xs[neg]] = [128, 128, 128]   # negative events: gray
        pos = ps > 0
        frame[ys[pos], xs[pos]] = [255, 255, 255]   # positive events: white
        return frame

    def slice_events(self, events):
        if self.slice_method == SliceMethod.FIRST_BY_TIME:
            min_time = events[:, 2].min()
            events = events[events[:, 2] < min_time + self.sample_len]
        elif self.slice_method == SliceMethod.LAST_BY_TIME:
            max_time = events[:, 2].max()
            events = events[events[:, 2] > max_time - self.sample_len]
        elif self.slice_method == SliceMethod.MID_BY_TIME:
            min_time = events[:, 2].min()
            max_time = events[:, 2].max()
            mid_time = (min_time + max_time) / 2
            half_len = self.sample_len / 2
            events = events[(events[:, 2] >= mid_time - half_len) & (events[:, 2] < mid_time + half_len)]
        elif self.slice_method == SliceMethod.FIRST_BY_COUNT:
            events = events[:self.num_events]
        elif self.slice_method == SliceMethod.LAST_BY_COUNT:
            events = events[-self.num_events:]
        elif self.slice_method == SliceMethod.MID_BY_COUNT:
            n = len(events)
            half = self.num_events // 2
            mid = n // 2
            events = events[mid - half:mid - half + self.num_events]
        return events
    
    def normalize_events(self, events):
        # Ensure polarity is -1/1
        events[:, 3] = torch.where(events[:, 3] >= 0, 1.0, -1.0)

        # Normalize to [0, 1] using sensor dims, then scale to norm targets
        events[:, 0] = (events[:, 0] / self.width * self.norm_w).floor()
        events[:, 1] = (events[:, 1] / self.height * self.norm_h).floor()
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        t_range = t_max - t_min
        if t_range > 0:
            events[:, 2] = ((events[:, 2] - t_min) / t_range * self.norm_t).floor()
        else:
            events[:, 2] = 0.0
        return events



class NCaltech101(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        data_cfg  = cfg.get("data",     {})
        train_cfg = cfg.get("training", {})

        self.data_dir    = Path(data_cfg["data_dir"])
        self.width       = data_cfg.get("sensor_width",  240)
        self.height      = data_cfg.get("sensor_height", 180)

        self.batch_size  = train_cfg.get("batch_size",  8)
        self.num_workers = train_cfg.get("num_workers", 4)

    @staticmethod
    def load_events(f_path: str, num_events: int):
        with h5py.File(str(f_path)) as fh:
            ev = fh['events']
            x = ev["x"][-num_events:]
            y = ev["y"][-num_events:]
            t = ev["t"][-num_events:]
            p = ev["p"][-num_events:]
        events = np.column_stack((x, y, t, p)).astype(np.float64)
        events[events[:, 3] == 0, 3] = -1
        return torch.from_numpy(events).float()

    @staticmethod
    def load_annotations(ann_file: str):
        f = open(ann_file)
        annotations = np.fromfile(f, dtype=np.int16)
        annotations = np.array(annotations[2:10])
        f.close()

        bbox = np.array([
            annotations[0], annotations[1],  # upper-left corner
            annotations[2] - annotations[0],  # width
            annotations[5] - annotations[1],  # height
        ])

        bbox[:2] = np.maximum(bbox[:2], 0)
        return bbox

    def _build_sample_list(self, split: str):
        """Scan data_dir/{split} and build list of (event_path, ann_path, class_idx) tuples."""
        events_dir = self.data_dir / split
        anns_dir   = self.data_dir / "annotations"

        class_names = sorted([
            d.name for d in events_dir.iterdir() if d.is_dir()
        ])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        samples = []
        for cls_name in class_names:
            cls_ev_dir  = events_dir / cls_name
            cls_ann_dir = anns_dir   / cls_name

            for ev_file in sorted(cls_ev_dir.glob("*.h5")):
                ann_file = cls_ann_dir / (ev_file.stem.replace("image_", "annotation_") + ".bin")
                if ann_file.exists():
                    samples.append((str(ev_file), str(ann_file), class_to_idx[cls_name]))

        return samples, class_to_idx

    def setup(self, stage=None):
        train_samples, class_to_idx = self._build_sample_list("training")
        val_samples,   _            = self._build_sample_list("validation")
        test_samples,  _            = self._build_sample_list("testing")

        self.train_data = NCaltech101Dataset(train_samples, class_to_idx, self.cfg, augment=True)
        self.val_data   = NCaltech101Dataset(val_samples,   class_to_idx, self.cfg)
        self.test_data  = NCaltech101Dataset(test_samples,  class_to_idx, self.cfg)

        self.num_classes  = len(class_to_idx)
        self.class_to_idx = class_to_idx

    def collate_fn(self, data_list):
        x = torch.cat([d.x for d in data_list], dim=0)
        pos = torch.cat([d.pos for d in data_list], dim=0)

        edge_index = []
        offset = 0
        for d in data_list:
            edge_index.append(d.edge_index + offset)
            offset += d.x.shape[0]
        edge_index = torch.cat(edge_index, dim=0)

        bbs = [d.bboxes for d in data_list]
        if any(bb.numel() for bb in bbs):
            bboxes = torch.cat(bbs, dim=0)
            batch_bb = torch.cat([
                torch.full((bb.size(0),), i, dtype=torch.long)
                for i, bb in enumerate(bbs)
            ], dim=0)
        else:
            bboxes = torch.empty((0, 5), dtype=torch.float32)
            batch_bb = torch.empty((0,), dtype=torch.long)


        batch = torch.cat([
            torch.full((d.x.shape[0],), i, dtype=torch.long)
            for i, d in enumerate(data_list)
        ], dim=0)

        target = convert_to_training_format(bboxes, batch_bb, batch.max().item()+1)

        frames = [d.frame for d in data_list]

        return GraphData(x=x,
                         pos=pos,
                         edge_index=edge_index,
                         batch=batch,
                         bboxes=bboxes,
                         batch_bb=batch_bb,
                         target=target,
                         frame=frames)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=self.collate_fn)


if __name__ == "__main__":
    ev_file  = "/home/imperator/Datasets/ncaltech101/training/accordion/image_0001.h5"
    ann_file = "/home/imperator/Datasets/ncaltech101/annotations/accordion/image_0001.bin"

    print(NCaltech101.load_events(ev_file, num_events=50000))
    print(NCaltech101.load_annotations(ann_file))
