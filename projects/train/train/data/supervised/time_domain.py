from typing import List, Optional

import torch

from train.augmentations import MultiRateResample
from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def __init__(
        self,
        *args,
        new_sample_rates: Optional[List[int]] = None,
        breakpoints: Optional[List[float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if new_sample_rates is not None and breakpoints is not None:
            self.resampler = MultiRateResample(
                original_sample_rate=self.hparams.sample_rate,
                duration=self.hparams.kernel_length,
                new_sample_rates=new_sample_rates,
                breakpoints=breakpoints,
            )
        else:
            self.resampler = None

    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        if self.resampler is not None:
            X_bg = self.resampler(X_bg)
            X_fg = self.resampler(X_fg)
        return X_bg, X_fg

    def augment(self, X, waveforms):
        X, y, psds = super().augment(X, waveforms)
        X = self.whitener(X, psds)
        if self.resampler is not None:
            X = self.resampler(X)
        return X, y
