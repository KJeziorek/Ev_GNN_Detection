import os
import numpy as np
import torch

import lightning as L

class NCaltech101(L.LightningDataModule):
    def __init__(self):
        super().__init__()

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


if __name__ == "__main__":

    raw_file = "/home/imperator/Datasets/NCaltech101/Caltech101/accordion/image_0001.bin"
    ann_file = "/home/imperator/Datasets/NCaltech101/Caltech101_annotations/accordion/annotation_0001.bin"

    ds = NCaltech101()
    print(ds.load_events(raw_file))
    print(ds.load_annotations(ann_file))