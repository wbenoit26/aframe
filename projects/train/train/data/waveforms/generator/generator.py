from typing import Callable, TYPE_CHECKING

from ..sampler import WaveformSampler
import torch
import h5py
from pathlib import Path


if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        val_waveform_file: Path,
        training_prior: Callable,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.

        Args:
            val_waveform_file:
                Path to the validation waveforms file.
            training_prior:
                A callable that returns a prior distribution
                for the parameters of the waveform generator.

        """
        super().__init__(*args, **kwargs)
        self.training_prior, _ = training_prior()
        self.val_waveform_file = val_waveform_file

        with h5py.File(val_waveform_file) as f:
            key = list(f["waveforms"].keys())[0]
            self.num_val_waveforms = len(f["waveforms"][key])

    def get_train_waveforms(self, *_):
        """
        Method is not implemented for this class, as
        waveforms are generated on the fly.
        """
        pass

    # TODO: Is there ever a reason to generate validation waveforms
    # on the fly, given that they need to be rejection-sampled?
    # Does is always make sense to load from disk?
    # def get_val_waveforms(self, world_size, _):
    #     N = self.num_val_waveforms // world_size
    #     parameters = self.training_prior.sample(N)
    #     generation_params = self.convert(parameters)
    #     generation_params = {
    #         k: torch.Tensor(v) for k, v in generation_params.items()
    #     }
    #     hc, hp = self(**generation_params)
    #     return hc, hp

    def sample(self, X: torch.Tensor):
        N = len(X)
        parameters = self.training_prior.sample(N)
        generation_params = self.convert(parameters)
        generation_params = {
            k: torch.Tensor(v).to(X.device)
            for k, v in generation_params.items()
        }
        hc, hp = self(**generation_params)
        return hc, hp

    def convert(self, parameters: dict) -> dict:
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
