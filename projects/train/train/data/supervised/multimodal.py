import torch
from ml4gw.utils.slicing import sample_kernels
from train.data.supervised.supervised import SupervisedAframeDataset


class MultiModalLocalizeDataset(SupervisedAframeDataset):
    def inject(self, X):
        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)

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
