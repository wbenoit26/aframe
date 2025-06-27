from typing import Callable, TYPE_CHECKING

from ..sampler import WaveformSampler
import torch

from ledger.injections import BilbyParameterSet

if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        num_val_waveforms: int,
        training_prior: Callable,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.

        Args:
            num_val_waveforms:
                Total number of validation waveforms to use.
                This total will be split up among all devices
            training_prior:
                A callable that returns a prior distribution
                for the parameters of the waveform generator.
                It should take no arguments and return an object
                with a `sample` method that takes an integer N.

        """
        super().__init__(*args, **kwargs)
        self.training_prior, _ = training_prior()
        self.num_val_waveforms = num_val_waveforms

    def get_train_waveforms(self, *_):
        """
        Method is not implemented for this class, as
        waveforms are generated on the fly.
        """
        pass

    def get_val_waveforms(self, world_size, _):
        N = self.num_val_waveforms // world_size
        parameters = self.training_prior.sample(N)
        parameter_set = BilbyParameterSet(**parameters)
        generation_params = parameter_set.generation_params(
            reference_frequency=40
        )
        generation_params = {
            k: torch.Tensor(v) for k, v in generation_params.items()
        }
        hc, hp = self(**generation_params)
        return hc, hp

    def sample(self, X: torch.Tensor):
        N = len(X)
        parameters = self.training_prior.sample(N)
        parameter_set = BilbyParameterSet(**parameters)
        generation_params = parameter_set.generation_params(
            reference_frequency=40
        )
        generation_params = {
            k: torch.Tensor(v).to(X.device)
            for k, v in generation_params.items()
        }
        hc, hp = self(**generation_params)
        return hc, hp

    def forward(self):
        raise NotImplementedError
