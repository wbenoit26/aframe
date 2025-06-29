from typing import Callable, TYPE_CHECKING

from ..sampler import WaveformSampler
import torch
import h5py
from pathlib import Path

from utils import x_per_y

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

    def get_slice_bounds(self, total, world_size, rank) -> tuple[int, int]:
        """
        Determine waveform indices to load for this device
        given our rank and world size
        """
        per_dev = x_per_y(abs(total), world_size)
        start = rank * per_dev
        stop = (rank + 1) * per_dev
        return start, stop

    def get_val_waveforms(self, world_size, rank):
        """
        Returns validation waveforms for this device
        """
        start, stop = self.get_slice_bounds(
            self.num_val_waveforms, world_size, rank
        )
        with h5py.File(self.val_waveform_file) as f:
            coalescence_time = f.attrs["coalescence_time"]
            duration = f.attrs["duration"]
            if self.right_pad != duration - coalescence_time:
                raise ValueError(
                    f"Right padding {self.right_pad} does not match duration "
                    f"{duration} - coalescence time {coalescence_time}"
                )
            waveforms = []
            for key in f["waveforms"].keys():
                waveforms.append(torch.Tensor(f["waveforms"][key][start:stop]))

        return torch.stack(waveforms, dim=0)

    # TODO: Is there ever a reason to generate validation waveforms
    # on the fly, given that they need to be rejection-sampled?
    # Does is always make sense to load from disk?
    # def get_val_waveforms(self, world_size, _):
    #     N = self.num_val_waveforms // world_size
    #     parameters = self.training_prior.sample(N)
    #     parameter_set = BilbyParameterSet(**parameters)
    #     generation_params = parameter_set.generation_params(
    #         reference_frequency=40
    #     )
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
