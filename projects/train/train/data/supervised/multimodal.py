import torch
from ml4gw.utils.slicing import sample_kernels
from train.data.supervised.supervised import SupervisedAframeDataset


class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, cross, plus):
        X_bg, X_inj, psds = super().build_val_batches(background, cross, plus)
        X_bg = self.whitener(X_bg, psds)
        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(
            X_bg.shape[-1], d=1 / self.hparams.sample_rate
        )
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg, psds

    def on_after_batch_transfer(self, batch, _):
        """
        This is a method inherited from the DataModule
        base class that gets called after data returned
        by a dataloader gets put on the local device,
        but before it gets passed to the LightningModule.
        Use this to do on-device augmentation/preprocessing.
        """
        if self.trainer.training:
            # if we're training, perform random augmentations
            # on input data and use it to impact labels
            [batch] = batch
            batch = self.inject(batch)
        elif self.trainer.validating or self.trainer.sanity_checking:
            # If we're in validation mode but we're not validating
            # on the local device, the relevant tensors will be
            # empty, so just pass them through with a 0 shift to
            # indicate that this should be ignored
            [background, _, timeslide_idx], [cross, plus] = batch

            # If we're validating, unfold the background
            # data into a batch of overlapping kernels now that
            # we're on the GPU so that we're not transferring as
            # much data from CPU to GPU. Once everything is
            # on-device, pre-inject signals into background.
            shift = self.timeslides[timeslide_idx].shift_size
            X_bg, X_fg, psds = self.build_val_batches(background, cross, plus)
            batch = (shift, X_bg, X_fg, psds)
        return batch

    def inject(self, X):
        X, y, psds = super().inject(X)
        X = self.whitener(X, psds)
        return (X, psds), y


class MultiModalLocalizeDataset(MultiModalSupervisedAframeDataset):
    def inject(self, X):
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

        # calculate asds and highpass
        freqs = torch.fft.rfftfreq(X.shape[-1], d=1 / self.hparams.sample_rate)
        num_freqs = len(freqs)
        psds = torch.nn.functional.interpolate(
            psds, size=(num_freqs,), mode="linear"
        )

        # sample enough waveforms to do true injections,
        # swapping, and muting
        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob

        dec, psi, phi = self.sample_extrinsic(X[mask])
        hc, hp = self.waveform_sampler.sample(X[mask])

        snrs = self.snr_sampler.sample((mask.sum().item(),)).to(X.device)
        responses = self.projector(
            dec, psi, phi, snrs, psds[mask], cross=hc, plus=hp
        )
        responses = self.slice_waveforms(responses)
        kernels, idx = sample_kernels(
            responses,
            kernel_size=X.size(-1),
            coincident=True,
            return_idx=True,
        )
        max_idx = (
            self.kernel_size - self.right_pad_size
        ) + self.filter_size // 2
        signal_idx = max_idx - idx
        signal_idx = signal_idx.float() / max_idx
        y_loc = torch.zeros((X.size(0), 1), device=X.device)
        y_loc[mask, 0] = signal_idx

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        swap_indices = mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y_class = torch.zeros((X.size(0), 1), device=X.device)
        y_class[mask] += 1
        y_loc[~mask] = -1

        return (X, psds), y_class, y_loc
