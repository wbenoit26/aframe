import torch

Distribution = torch.distributions.Distribution


class WaveformSampler(torch.nn.Module):
    """
    Base object defining methods that waveform producing classes
    should implement. Should not be instantiated on its own.

    Args:
        fduration:
            Desired length in seconds of the time domain
            response of the whitening filter built from PSDs.
            See `ml4gw.spectral.truncate_inverse_power_spectrum`
        kernel_length:
            Length in seconds of window passed to neural network.
        sample_rate:
            Sample rate in Hz of generated waveforms

    """

    def __init__(
        self,
        *args,
        fduration: float,
        kernel_length: float,
        sample_rate: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fduration = fduration
        self.kernel_length = kernel_length
        self.sample_rate = sample_rate

    @property
    def duration(self):
        """
        Length of kernel before whitening removes
        fduration / 2 from each side
        """
        return self.fduration + self.kernel_length

    def get_val_waveforms(self):
        raise NotImplementedError

    def get_test_waveforms(self):
        raise NotImplementedError

    def sample(self, X: torch.Tensor):
        """Defines how to sample waveforms for training"""
        raise NotImplementedError
