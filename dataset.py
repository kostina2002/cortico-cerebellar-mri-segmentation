import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import nrrd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import cfg

class AtriaSegDataset(Dataset):
    def __init__(self, base_path, split='train', target_size=128, max_slices=96,
                 slice_step=1, random_state=42):
        self.base_path = Path(base_path)
        self.target_size = target_size
        self.max_slices = max_slices
        self.slice_step = slice_step
        self.augment = (split == 'train')
        self.samples = {}

        for sample_dir in self.base_path.iterdir():
            if not sample_dir.is_dir():
                continue
            files = {'mri': None, 'mask': None}
            for f in sample_dir.glob('*.nrrd'):
                fname = f.name.lower()
                if 'lgemri' in fname:
                    files['mri'] = str(f)
                elif 'laendo' in fname:
                    files['mask'] = str(f)
            if files['mri'] and files['mask']:
                self.samples[sample_dir.name] = files

        sample_ids = list(self.samples.keys())
        train_ids, val_ids = train_test_split(sample_ids, test_size=cfg.VAL_SPLIT,
                                              random_state=random_state)
        self.sample_ids = train_ids if split == 'train' else val_ids
        print(f"{split}: {len(self.sample_ids)} samples (slice_step={slice_step})")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        mri, _ = nrrd.read(self.samples[sid]['mri'])
        mask, _ = nrrd.read(self.samples[sid]['mask'])
        mask = (mask > 0).astype(np.float32)

        # Intensity normalization: clip to 99th percentile and scale to [0,1]
        p99 = np.percentile(mri[mri > 0], 99)
        mri = np.clip(mri / (p99 + 1e-8), 0, 1.0).astype(np.float32)

        depth = mri.shape[0]
        if depth > self.max_slices:
            start = (depth - self.max_slices) // 2
            mri = mri[start:start+self.max_slices]
            mask = mask[start:start+self.max_slices]
        elif depth < self.max_slices:
            pad = self.max_slices - depth
            mri = np.pad(mri, ((0,pad),(0,0),(0,0)), mode='constant')
            mask = np.pad(mask, ((0,pad),(0,0),(0,0)), mode='constant')

        mri_slices, mask_slices = [], []
        for i in range(self.max_slices):
            m_slice = torch.from_numpy(mri[i]).unsqueeze(0).unsqueeze(0)
            m_slice = F.interpolate(m_slice, size=(self.target_size, self.target_size),
                                    mode='bilinear', align_corners=False).squeeze(0)
            k_slice = torch.from_numpy(mask[i]).unsqueeze(0).unsqueeze(0)
            k_slice = F.interpolate(k_slice, size=(self.target_size, self.target_size),
                                    mode='nearest').squeeze(0)


            # Apply augmentations only during training and with configured probability
            if self.augment and random.random() < cfg.AUGMENTATION_PROB:
                # Add Gaussian noise
                noise = torch.randn_like(m_slice) * cfg.NOISE_STD
                m_slice = m_slice + noise
                
                # Random spatial shift
                shift_x = random.randint(-cfg.SHIFT_MAX, cfg.SHIFT_MAX)
                shift_y = random.randint(-cfg.SHIFT_MAX, cfg.SHIFT_MAX)
                m_slice = torch.roll(m_slice, shifts=(shift_x, shift_y), dims=(1,2))
                k_slice = torch.roll(k_slice, shifts=(shift_x, shift_y), dims=(1,2))
                
                # Zero-pad out-of-bound regions
                if shift_x > 0:
                    m_slice[:, :shift_x, :] = 0
                    k_slice[:, :shift_x, :] = 0
                elif shift_x < 0:
                    m_slice[:, shift_x:, :] = 0
                    k_slice[:, shift_x:, :] = 0
                if shift_y > 0:
                    m_slice[:, :, :shift_y] = 0
                    k_slice[:, :, :shift_y] = 0
                elif shift_y < 0:
                    m_slice[:, :, shift_y:] = 0
                    k_slice[:, :, shift_y:] = 0

            mri_slices.append(m_slice)
            mask_slices.append(k_slice)

        # Slice subsampling
        if self.slice_step > 1:
            indices = list(range(0, len(mri_slices), self.slice_step))
            mri_slices = [mri_slices[i] for i in indices]
            mask_slices = [mask_slices[i] for i in indices]

        return torch.stack(mri_slices), torch.stack(mask_slices)
