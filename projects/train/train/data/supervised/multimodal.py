import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class MultiModalSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, cross, plus):
        X_bg, X_inj, psds = super().build_val_batches(background, cross, plus)
        X_bg = self.whitener(X_bg, psds)
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
