from typing import TYPE_CHECKING


from ....prior import AmplfiPrior
from ..sampler import WaveformSampler

if TYPE_CHECKING:
    pass


class WaveformGenerator(WaveformSampler):
    def __init__(
        self,
        *args,
        num_val_waveforms: int,
        training_prior: AmplfiPrior,
        num_fit_params: int,
        **kwargs,
    ):
        """
        A torch module for generating waveforms on the fly.

        Args:
            num_val_waveforms:
                Total number of validation waveforms to use.
                This total will be split up among all devices
            training_prior:
                A callable that takes an integer N and
                returns a dictionary of parameter Tensors, each of length `N`

        """
        super().__init__(*args, **kwargs)
        self.training_prior = training_prior
        self.num_val_waveforms = num_val_waveforms
        self.num_fit_params = num_fit_params

    def get_val_waveforms(self, _, world_size):
        num_waveforms = self.num_val_waveforms // world_size
        parameters = self.training_prior(num_waveforms, device="cpu")
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def sample(self, X):
        N = len(X)
        parameters = self.training_prior(N, device=X.device)
        hc, hp = self(**parameters)
        return hc, hp, parameters

    def forward(self):
        raise NotImplementedError
