import os
import numpy as np
import torch
import lightning as L

from pathlib import Path
from torch.utils.data import Dataset, DataLoader


from utils.data import GraphData
from datasets.graph_gen.matrix_neighbour import GraphGenerator


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
            cfg: config dict
            augment: whether to apply data augmentation
        """
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.cfg = cfg
        self.augment = augment

        self.width = cfg.get("sensor_width", 240)
        self.height = cfg.get("sensor_height", 180)
        self.num_events = cfg.get("num_events", 50000)

        self.generator = GraphGenerator(width=self.width, height=self.height)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ev_path, ann_path, class_idx = self.samples[idx]

        events = NCaltech101.load_events(ev_path)
        bbox = NCaltech101.load_annotations(ann_path)

        # Limit number of events
        events = events[:self.num_events]

        # bbox is [x, y, w, h] -> prepend class_id to get [class_id, x, y, w, h]
        bbox = np.concatenate([[class_idx], bbox]).astype(np.float32)
        bbox = bbox.reshape(1, 5)  # (1, 5) — single bbox per sample

        # Generate graph
        x, pos, edge_index = self.generator.generate_edges(
            events,
            radius_x=self.cfg.get("radius_x", 5),
            radius_y=self.cfg.get("radius_y", 5),
            radius_t=self.cfg.get("radius_t", 0.001),
        )

        self.generator.clear()

        bboxes = torch.from_numpy(bbox).to(dtype=torch.float32)
        
        return GraphData(x=x, pos=pos, edge_index=edge_index, bboxes=bboxes)


class NCaltech101(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg["data_dir"])
        self.width = cfg.get("sensor_width", 240)
        self.height = cfg.get("sensor_height", 180)

        self.train_ratio = cfg.get("train_ratio", 0.7)
        self.val_ratio = cfg.get("val_ratio", 0.15)

    @staticmethod
    def load_events(raw_file: str):
        f = open(raw_file, 'rb')
        raw_data = np.fromfile(f, dtype=np.uint8)
        f.close()

        raw_data = np.uint32(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7  # bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        all_ts = all_ts / 1e6  # µs -> s
        all_p = all_p.astype(np.float64)
        all_p[all_p == 0] = -1
        events = np.column_stack((all_x, all_y, all_ts, all_p))
        events = torch.from_numpy(events).float()
        return events

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

    def _build_sample_list(self):
        """Scan data_dir and build list of (event_path, ann_path, class_idx) tuples."""
        events_dir = self.data_dir / "Caltech101"
        anns_dir = self.data_dir / "Caltech101_annotations"

        class_names = sorted([
            d.name for d in events_dir.iterdir()
            if d.is_dir() and d.name != "BACKGROUND_Google"
        ])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        samples = []
        for cls_name in class_names:
            cls_ev_dir = events_dir / cls_name
            cls_ann_dir = anns_dir / cls_name

            for ev_file in sorted(cls_ev_dir.glob("*.bin")):
                # image_0001.bin -> annotation_0001.bin
                ann_file = cls_ann_dir / ev_file.name.replace("image_", "annotation_")
                if ann_file.exists():
                    samples.append((str(ev_file), str(ann_file), class_to_idx[cls_name]))

        return samples, class_to_idx

    def setup(self, stage=None):
        samples, class_to_idx = self._build_sample_list()

        # Deterministic shuffle for reproducible splits
        rng = np.random.RandomState(seed=self.cfg.get("seed", 42))
        rng.shuffle(samples)

        n = len(samples)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train + n_val]
        test_samples = samples[n_train + n_val:]

        self.train_data = NCaltech101Dataset(train_samples, class_to_idx, self.cfg, augment=True)
        self.val_data = NCaltech101Dataset(val_samples, class_to_idx, self.cfg)
        self.test_data = NCaltech101Dataset(test_samples, class_to_idx, self.cfg)

        self.num_classes = len(class_to_idx)
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

        return GraphData(x=x, 
                         pos=pos,
                         edge_index=edge_index,
                         batch=batch,
                         bboxes=bboxes,
                         batch_bb=batch_bb)

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.cfg.get('batch_size', 32),
                          num_workers=self.cfg.get('num_workers', 16),
                          shuffle=True,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.cfg.get('batch_size', 32),
                          num_workers=self.cfg.get('num_workers', 16),
                          shuffle=False,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.cfg.get('batch_size', 32),
                          num_workers=self.cfg.get('num_workers', 16),
                          shuffle=False,
                          collate_fn=self.collate_fn)


if __name__ == "__main__":
    raw_file = "/home/imperator/Datasets/NCaltech101/Caltech101/accordion/image_0001.bin"
    ann_file = "/home/imperator/Datasets/NCaltech101/Caltech101_annotations/accordion/annotation_0001.bin"

    ds = NCaltech101(cfg={"data_dir": "/home/imperator/Datasets/NCaltech101"})
    print(ds.load_events(raw_file))
    print(ds.load_annotations(ann_file))
