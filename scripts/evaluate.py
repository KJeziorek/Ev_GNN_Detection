import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets.graph_gen.matrix_neighbour import GraphGenerator
from datasets.ncaltech101 import NCaltech101
from models.backbone import BACKBONE

raw_file = "/home/imperator/Datasets/NCaltech101/Caltech101/accordion/image_0001.bin"
ann_file = "/home/imperator/Datasets/NCaltech101/Caltech101_annotations/accordion/annotation_0001.bin"


gen = GraphGenerator(width=240, height=180)  # NCaltech101 resolution

ds = NCaltech101()
layer = BACKBONE().to('cuda')
events = ds.load_events(raw_file)
ann = ds.load_annotations(ann_file)


from time import time

for i in range(100):
    t0 = time()
    X, P, E = gen.generate_edges(events, radius_x=5, radius_y=5, radius_t=0.001)
    gen.clear()
    out = layer(X.to('cuda'), P.to('cuda'), E.to('cuda'))
    t1 = time()
    print(t1 - t0, out)
