import numpy as np
import h5py
from scipy.ndimage import zoom, gaussian_filter

class SliceDS:
    def __init__(self, path, length, rng, crop=False, resample=None):
        self.path = path
        self.rng = rng
        self.crop = crop
        self.resample = resample
        self.length = length
        self.file = None

    def __len__(self):
        return self.length
        #if self.file is None:
        #    self.file = h5py.File(self.path, 'r')
        #return self.file["dataset"].shape[0]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')

        src_slice = self.file["dataset"][index].reshape(512, 512)

        #src_slice, tgt_slice = np.split(chunk, 2, axis=0)
        #src_slice = src_slice.reshape(512, 512)
        #tgt_slice = tgt_slice.reshape(512, 512)

        #src_slice = (np.clip(src_slice, -50, 350) + 50) / 400
        #tgt_slice = (np.clip(tgt_slice, -50, 350) + 50) / 400

        src_slice = np.clip(src_slice, 0, 100) / 100
        #tgt_slice = np.clip(tgt_slice, 0, 100) / 100

        seg_slice = self.file["segmentations"][index].reshape(512, 512)

        if self.crop:
            h = src_slice.shape[0]
            w = src_slice.shape[1]
            pad_h = 256 - h if h < 256 else 0
            pad_w = 256 - w if w < 256 else 0
            if pad_h > 0 or pad_w > 0:
                src_slice = np.pad(src_slice, ((0,pad_h), (0, pad_w)))
                tgt_slice = np.pad(tgt_slice, ((0,pad_h), (0, pad_w)))
                h = src_slice.shape[0]
                w = src_slice.shape[1]

            x = self.rng.integers(0, h-256) if h-256 > 0 else 0
            y = self.rng.integers(0, w-256) if w-256 > 0 else 0

            src_slice = src_slice[x:x+256, y:y+256]
            #tgt_slice = tgt_slice[x:x+256, y:y+256]
            seg_slice = seg_slice[x:x+256, y:y+256]

        if self.resample is not None:
            order = 1
            prefilter = False
            #target_shape = (256, 256)
            #target_shape = (128, 128)
            target_shape = self.resample
            dsfactor = [w / float(f) for w, f in zip(target_shape, src_slice.shape)]
            src_slice = zoom(src_slice, zoom=dsfactor, order=order, prefilter=prefilter)
            seg_slice = zoom(seg_slice, zoom=dsfactor, order=0, prefilter=False)

        src_slice = np.expand_dims(src_slice, -1)
        #tgt_slice = np.expand_dims(tgt_slice, -1)
        seg_slice = np.expand_dims(seg_slice, -1)

        seg_slice = gaussian_filter(seg_slice, sigma=6)
        seg_slice = np.where(seg_slice > 0.1, 1, 0)

        #return src_slice#, tgt_slice
        return src_slice, seg_slice


class SingleSliceDS:
    def __init__(self, path, length, rng, crop=False, resample=None):
        self.path = path
        self.rng = rng
        self.crop = crop
        self.resample = resample
        self.length = length
        self.file = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, 'r')

        src_slice = self.file["dataset"][index].reshape(512, 512)
        #src_slice = (np.clip(src_slice, -50, 350) + 50) / 400
        src_slice = np.clip(src_slice, 0, 100) / 100

        if self.crop:
            h = src_slice.shape[0]
            w = src_slice.shape[1]
            pad_h = 256 - h if h < 256 else 0
            pad_w = 256 - w if w < 256 else 0
            if pad_h > 0 or pad_w > 0:
                src_slice = np.pad(src_slice, ((0,pad_h), (0, pad_w)))
                h = src_slice.shape[0]
                w = src_slice.shape[1]

            x = self.rng.integers(0, h-256) if h-256 > 0 else 0
            y = self.rng.integers(0, w-256) if w-256 > 0 else 0

            src_slice = src_slice[x:x+256, y:y+256]

        if self.resample is not None:
            order = 1
            prefilter = False
            #target_shape = (64, 64)
            #target_shape = (128, 128)
            target_shape = self.resample
            #target_shape = (256, 256)
            dsfactor = [w / float(f) for w, f in zip(target_shape, src_slice.shape)]
            src_slice = zoom(src_slice, zoom=dsfactor, order=order, prefilter=prefilter)
        
        src_slice = np.expand_dims(src_slice, -1)

        #order = 1
        #prefilter = False
        #low_target_shape = (64, 64)
        #high_target_shape = (256, 256)
        #low_dsfactor = [w / float(f) for w, f in zip(low_target_shape, src_slice.shape)]
        #high_dsfactor = [w / float(f) for w, f in zip(high_target_shape, src_slice.shape)]
        #low_src_slice = zoom(src_slice, zoom=low_dsfactor, order=order, prefilter=prefilter)
        #high_src_slice = zoom(src_slice, zoom=high_dsfactor, order=order, prefilter=prefilter)

        #low_src_slice = np.expand_dims(low_src_slice, -1)
        #high_src_slice = np.expand_dims(high_src_slice, -1)

        #return low_src_slice, high_src_slice
        return src_slice
