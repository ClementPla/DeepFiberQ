from skimage.measure import label

from skimage.morphology import skeletonize
from skimage.segmentation import relabel_sequential
import numpy as np


class PostProcessor:
    def __init__(self, predicted_mask):
        self.mask = np.zeros_like(predicted_mask)
        self.mask[predicted_mask == 1] = 1
        self.mask[predicted_mask == 2] = 2

        self._cc = None
        self._ncc = None

    @property
    def cc(self):
        if self._cc is None:
            self._cc, self._ncc = label(self.mask > 0, return_num=True)
        return self._cc

    @property
    def ncc(self):
        if self._ncc is None:
            self._cc, self._ncc = label(self.mask > 0, return_num=True)
        return self._ncc

    @ncc.setter
    def ncc(self, value):
        self._ncc = value

    def filter_by_size(self, min_size=100):
        labels, counts = np.unique(
            self.cc,
            return_counts=True,
        )
        for l, c in zip(labels, counts):
            if c < min_size:
                self.mask[self.cc == l] = 0
                self.cc[self.cc == l] = 0
                self._cc = relabel_sequential(self.cc)[0]
                self.ncc -= 1

    def filter_cc_if_contains_only_value(self):
        labels = np.unique(self.cc, return_counts=False)
        for l in labels:
            if np.unique(self.mask[self.cc == l]).size == 1:
                self.mask[self.cc == l] = 0
                self.cc[self.cc == l] = 0
                self._cc = relabel_sequential(self.cc)[0]
                self.ncc -= 1

    def filter_alternating_cc(self):
        labels = np.unique(self.cc, return_counts=False)
        for l in labels:
            if l == 0:
                continue
            values = self.mask[self.cc == l]
            if values.size < 3:
                continue
            diff = np.diff(values, n=1) > 0

            if sum(diff) > 1:
                self.mask[self.cc == l] = 0

        self._cc = None
        self.ncc = None

    def apply(self):
        self.filter_by_size(min_size=25)
        self.filter_cc_if_contains_only_value()

        skeleton = skeletonize(self.mask > 0)

        self.mask = self.mask * skeleton
        self._cc = None
        self.ncc = None
        self.filter_alternating_cc()

        return self.mask

    def count_ratio(self):
        strands = dict()
        for l in range(1, self.ncc + 1):
            current_strand = self.cc == l
            if not np.any(current_strand):
                continue

            x, y = np.where(current_strand)
            values = self.mask[current_strand]

            strands[l] = {
                "red": np.sum(values == 1),
                "green": np.sum(values == 2),
                "x": x,
                "y": y,
            }
        return strands
