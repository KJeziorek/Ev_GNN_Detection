import torch
import numpy as np
import numba
from typing import List, Tuple, Optional

################################################################################
#                               Low-level helpers                              #
################################################################################

@numba.njit
def _add_event(x: float, y: float, xlim: int, ylim: int,
               p: float, i: int, count: np.ndarray,
               pos: np.ndarray, mask: np.ndarray, threshold: float = 1.0):
    """Accumulate polarity in a 2x2 bilinear kernel and emit an event when the
    accumulated value crosses *threshold* (DVS subsampling heuristic)."""
    count[ylim, xlim] += p * (1.0 - abs(x - xlim)) * (1.0 - abs(y - ylim))
    pol = 1.0 if count[ylim, xlim] > 0.0 else -1.0

    if pol * count[ylim, xlim] > threshold:
        count[ylim, xlim] -= pol * threshold
        mask[i] = True
        pos[i, 0] = xlim
        pos[i, 1] = ylim


@numba.njit
def _subsample(pos: np.ndarray, polarity: np.ndarray, mask: np.ndarray,
               count: np.ndarray, threshold: float = 1.0):
    """Bilinear spatio-temporal subsampling of events (see _add_event)."""
    for i in range(pos.shape[0]):
        x, y = pos[i]
        x0, x1 = int(x), int(x + 1)
        y0, y1 = int(y), int(y + 1)

        _add_event(x, y, x0, y0, polarity[i, 0], i, count, pos, mask, threshold)
        _add_event(x, y, x1, y0, polarity[i, 0], i, count, pos, mask, threshold)
        _add_event(x, y, x0, y1, polarity[i, 0], i, count, pos, mask, threshold)
        _add_event(x, y, x1, y1, polarity[i, 0], i, count, pos, mask, threshold)

################################################################################
#                                Torch utils                                   #
################################################################################

def _scale_and_clip(x: float, scale: int) -> int:
    return int(max(0, min(x * scale, scale - 1)))


def _crop_events(events: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    mask = (
        (events[:, 0] >= left[0]) & (events[:, 0] <= right[0]) &
        (events[:, 1] >= left[1]) & (events[:, 1] <= right[1])
    )
    return events[mask]


def _crop_bbox(bbox: Optional[torch.Tensor], left: torch.Tensor, right: torch.Tensor) -> Optional[torch.Tensor]:
    if bbox is None or bbox.numel() == 0:
        return bbox

    bb = bbox.clone().float()

    x1, y1 = bb[:, 0], bb[:, 1]
    x2, y2 = x1 + bb[:, 2], y1 + bb[:, 3]

    x1 = x1.clamp(left[0], right[0])
    y1 = y1.clamp(left[1], right[1])
    x2 = x2.clamp(left[0], right[0])
    y2 = y2.clamp(left[1], right[1])

    w, h = x2 - x1, y2 - y1
    keep = (w > 0) & (h > 0)
    out = torch.stack([x1, y1, w, h, bb[:, 4]], dim=-1)
    return out[keep]

################################################################################
#                               Base class                                     #
################################################################################

class BaseTransform:
    def __call__(self, events: torch.Tensor,
                 bbox: Optional[torch.Tensor] = None):
        raise NotImplementedError

################################################################################
#                              Transforms                                      #
################################################################################

class RandomHFlip(BaseTransform):
    def __init__(self, p: float, width: int = 240):
        self.p = p
        self.width = width

    def __call__(self, events, bbox=None):
        if torch.rand(1).item() > self.p:
            return events, bbox

        ev = events.clone()
        ev[:, 0] = self.width - 1 - ev[:, 0]

        bb = None
        if bbox is not None:
            bb = bbox.clone().float()
            bb[:, 0] = self.width - 1 - (bb[:, 0] + bb[:, 2])
        return ev, bb


class Crop(BaseTransform):
    def __init__(self, min_: List[float], max_: List[float],
                 width: int = 240, height: int = 180):
        sz = [width, height]
        self.left = torch.tensor([_scale_and_clip(m, s) for m, s in zip(min_, sz)], dtype=torch.float32)
        self.right = torch.tensor([_scale_and_clip(m, s) for m, s in zip(max_, sz)], dtype=torch.float32)

    def __call__(self, events, bbox=None):
        ev = _crop_events(events, self.left, self.right)
        bb = _crop_bbox(bbox, self.left, self.right) if bbox is not None else None
        return ev, bb


class RandomZoom(BaseTransform):
    def __init__(self, zoom: Tuple[float, float], subsample: bool = True,
                 width: int = 240, height: int = 180):
        assert zoom[0] > 0 and zoom[1] >= zoom[0]
        self.zoom = zoom
        self.subsample = subsample
        self.width = width
        self.height = height
        self._count = None

    def init(self, height: int, width: int):
        self._count = np.zeros((height + 1, width + 1), dtype=np.float32)

    def _subsample_events(self, events: torch.Tensor, z: float) -> torch.Tensor:
        if self._count is None:
            self.init(self.height, self.width)
        count = self._count * 0.0

        events_np = events.numpy()
        pos = events_np[:, :2].astype(np.float32)
        pol = events_np[:, 3:4].astype(np.float32)
        mask = np.zeros(len(events_np), dtype=np.bool_)

        _subsample(pos, pol, mask, count, threshold=1.0 / (z ** 2))

        out = events[mask].clone()
        out[:, 0:2] = torch.from_numpy(pos[mask].astype(np.float32))
        return out

    def __call__(self, events, bbox=None):
        z = float(torch.empty(1).uniform_(self.zoom[0], self.zoom[1]).item())
        cx, cy = self.width // 2, self.height // 2

        ev = events.clone()
        ev[:, 0] = ((ev[:, 0] - cx) * z + cx).to(torch.int16).float()
        ev[:, 1] = ((ev[:, 1] - cy) * z + cy).to(torch.int16).float()

        if self.subsample and z < 1.0:
            ev = self._subsample_events(ev, z)

        bb = None
        if bbox is not None:
            bb = bbox.clone().float()
            bb[:, 2:4] *= z
            bb[:, 0] = (bb[:, 0] - cx) * z + cx
            bb[:, 1] = (bb[:, 1] - cy) * z + cy
        return ev, bb


class RandomCrop(BaseTransform):
    def __init__(self, size: List[float] = (0.75, 0.75), p: float = 0.5,
                 width: int = 240, height: int = 180):
        self.p = p
        img_sz = torch.tensor([width, height], dtype=torch.float32)
        self.size = torch.tensor([_scale_and_clip(s, ss) for s, ss in zip(size, [width, height])], dtype=torch.float32)
        self.left_max = img_sz - self.size

    def __call__(self, events, bbox=None):
        if torch.rand(1).item() > self.p:
            return events, bbox

        left = (torch.rand(2) * self.left_max).to(torch.int16).float()
        right = left + self.size

        ev = _crop_events(events, left, right)
        bb = _crop_bbox(bbox, left, right) if bbox is not None else None
        return ev, bb


class RandomTranslate(BaseTransform):
    def __init__(self, size: List[float], width: int = 240, height: int = 180):
        img_sz = [width, height]
        self.size_px = torch.tensor([_scale_and_clip(s, ss) for s, ss in zip(size[:2], img_sz)], dtype=torch.float32)

    def __call__(self, events, bbox=None):
        move = ((torch.rand(2) * 2.0 - 1.0) * self.size_px).to(torch.int16).float()

        ev = events.clone()
        ev[:, 0:2] += move

        bb = None
        if bbox is not None:
            bb = bbox.clone().float()
            bb[:, 0:2] += move
        return ev, bb

################################################################################
#                             Compose helper                                   #
################################################################################

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, bbox=None):
        for t in self.transforms:
            events, bbox = t(events, bbox)
        return events, bbox

    def init(self, height, width):
        for t in self.transforms:
            if hasattr(t, "init"):
                t.init(height, width)
