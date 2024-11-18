from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch
from gwpy.timeseries import TimeSeries
from torchaudio.transforms import Resample

from ledger.injections import InterferometerResponseSet, waveform_class_factory
from plots.vizapp.infer.utils import get_indices, get_strain_fname
from utils.preprocessing import BackgroundSnapshotter, BatchWhitener


class MultiRateResample(torch.nn.Module):
    """
    Resample a time series to multiple different sample rates

    Args:
        original_sample_rate:
            The sample rate of the original time series in Hz
        duration:
            The duration of the original time series in seconds
        new_sample_rates:
            A list of new sample rates that different portions
            of the time series will be resampled to
        breakpoints:
            The time at which there is a transition from one
            sample rate to another

    Returns:
        A time series Tensor with each of the resampled segments
        concatenated together
    """

    def __init__(
        self,
        original_sample_rate: int,
        duration: float,
        new_sample_rates: List[int],
        breakpoints: List[float],
    ):
        super().__init__()
        self.original_sample_rate = original_sample_rate
        self.duration = duration
        self.new_sample_rates = new_sample_rates
        self.breakpoints = breakpoints
        self._validate_inputs()

        # Add endpoints to breakpoint list
        self.breakpoints.append(duration)
        self.breakpoints.insert(0, 0)

        self.resamplers = torch.nn.ModuleList(
            [Resample(original_sample_rate, new) for new in new_sample_rates]
        )
        idxs = [
            [int(breakpoints[i] * new), int(breakpoints[i + 1] * new)]
            for i, new in enumerate(self.new_sample_rates)
        ]
        self.register_buffer("idxs", torch.Tensor(idxs).int())

    def _validate_inputs(self):
        if len(self.new_sample_rates) != len(self.breakpoints) + 1:
            raise ValueError(
                "There are too many/few breakpoints given "
                "for the number of frequencies"
            )
        if max(self.breakpoints) >= self.duration:
            raise ValueError(
                "At least one breakpoint was greater than the given duration"
            )
        if not self.breakpoints[1:] > self.breakpoints[:-1]:
            raise ValueError("Breakpoints must be sorted in ascending order")

    def forward(self, X: torch.Tensor):
        X = X.contiguous()
        return torch.cat(
            [
                resample(X)[..., idx[0] : idx[1]]
                for resample, idx in zip(self.resamplers, self.idxs)
            ],
            dim=-1,
        )


class EventAnalyzer:
    """
    Class for performing on-the-fly inference
    """

    def __init__(
        self,
        model: torch.nn.Module,
        strain_dir: Path,
        response_set: Path,
        psd_length: float,
        kernel_length: float,
        sample_rate: float,
        fduration: float,
        inference_sampling_rate: float,
        integration_length: float,
        batch_size: int,
        highpass: float,
        fftlength: float,
        device: str,
        ifos: List[str],
        padding: int = 3,
    ):
        self.model = model
        augmentor = MultiRateResample(
            original_sample_rate=sample_rate,
            duration=kernel_length,
            new_sample_rates=[128, 256, 512, 1024, 2048],
            breakpoints=[4, 5, 5.5, 5.75],
        )
        self.whitener = BatchWhitener(
            kernel_length,
            sample_rate,
            inference_sampling_rate,
            batch_size,
            fduration,
            fftlength=fftlength,
            highpass=highpass,
            augmentor=augmentor,
            return_whitened=True,
        ).to(device)

        self.snapshotter = BackgroundSnapshotter(
            psd_length=psd_length,
            kernel_length=kernel_length,
            fduration=fduration,
            sample_rate=sample_rate,
            inference_sampling_rate=inference_sampling_rate,
        ).to(device)

        self.response_set = response_set
        self.strain_dir = strain_dir
        self.ifos = ifos
        self.padding = padding
        self.sample_rate = sample_rate
        self.fduration = fduration
        self.psd_length = psd_length
        self.kernel_length = kernel_length
        self.highpass = highpass
        self.inference_sampling_rate = inference_sampling_rate
        self.integration_length = integration_length
        self.batch_size = batch_size
        self.device = device

    @property
    def waveform_class(self):
        return waveform_class_factory(
            self.ifos, InterferometerResponseSet, "IfoWaveformSet"
        )

    @property
    def kernel_size(self):
        return int(self.kernel_length * self.sample_rate)

    @property
    def state_shape(self):
        return (1, len(self.ifos), self.snapshotter.state_size)

    @property
    def inference_stride(self):
        return int(self.sample_rate / self.inference_sampling_rate)

    @property
    def step_size(self):
        return int(self.batch_size * self.inference_stride)

    @property
    def integration_size(self):
        return int(self.integration_length * self.inference_sampling_rate)

    @property
    def window(self):
        return np.ones((self.integration_size,)) / self.integration_size

    @property
    def times(self):
        """
        Returns the time values relative to event time
        """
        start = (
            self.psd_length
            + self.kernel_length
            + (self.fduration / 2)
            + self.padding
        )
        stop = self.kernel_length + (self.fduration / 2) + self.padding
        return np.arange(-start, stop, 1 / self.sample_rate)

    @property
    def inference_times(self):
        return self.times[:: self.inference_stride]

    @property
    def whitened_times(self):
        start = (
            self.step_size
            - self.inference_stride
            - int(self.sample_rate * self.fduration)
        )
        return self.times[start:]

    def find_strain(self, time: float, shifts: Sequence[float]):
        # find strain file corresponding to requested time
        fname, t0, duration = get_strain_fname(self.strain_dir, time)
        # find indices of data needed for inference
        times = np.arange(t0, t0 + duration, 1 / self.sample_rate)
        start, stop = get_indices(
            times, time + self.times[0], time + self.times[-1]
        )
        strain = []
        with h5py.File(fname, "r") as f:
            for ifo, shift in zip(self.ifos, shifts):
                shift_size = int(shift * self.sample_rate)
                start_shifted, stop_shifted = (
                    start + shift_size,
                    stop + shift_size,
                )
                data = torch.tensor(f[ifo][start_shifted:stop_shifted])
                strain.append(data)

        return torch.stack(strain, axis=0), time + self.times[0]

    def find_waveform(self, time: float, shifts: np.ndarray):
        """
        find the closest injection that corresponds to event
        time and shifts from waveform dataset
        """
        waveform = self.waveform_class.read(
            self.response_set, time - 0.1, time + 0.1, shifts
        )
        return waveform

    def integrate(self, y):
        if self.integration_size == 0:
            return y
        integrated = np.convolve(y, self.window, mode="full")
        return integrated[: -self.integration_size + 1]

    def infer(self, X: torch.Tensor):
        ys, strain = [], []
        start = 0
        state = torch.zeros(self.state_shape).to(self.device)
        # pad X up to batch size
        remainder = X.shape[-1] % self.step_size
        num_slice = None
        if remainder:
            pad = self.step_size - remainder
            X = torch.nn.functional.pad(X, (0, pad))
            num_slice = pad // self.inference_stride
        slc = slice(-num_slice)

        while start <= (X.shape[-1] - self.step_size):
            stop = start + self.step_size
            x = X[:, :, start:stop]
            with torch.no_grad():
                x, state = self.snapshotter(x, state)
                batch, whitened = self.whitener(x)
                y_hat = self.model(batch)[:, 0].cpu().numpy()

            strain.append(whitened.cpu().numpy())
            ys.append(y_hat)
            start += self.step_size

        whitened = np.concatenate(strain, axis=-1)[..., :-pad]
        ys = np.concatenate(ys)[slc]
        return ys, whitened

    def analyze(self, time, shifts, foreground):
        strain, t0 = self.find_strain(time, shifts)
        if foreground:
            waveform = self.find_waveform(time, shifts)
            strain = waveform.inject(strain, t0)
        strain = strain[None]
        strain = torch.Tensor(strain).to(self.device)
        nn, whitened = self.infer(strain)
        integrated = self.integrate(nn)

        return nn, integrated, whitened

    def get_fft(self, strain: Dict[str, np.ndarray]):
        ffts = {}
        for ifo in self.ifos:
            data = strain[ifo]
            ts = TimeSeries(data, times=self.whitened_times)
            ts = ts.crop(-3, 5)
            fft = ts.fft().crop(start=self.highpass)
            freqs = fft.frequencies.value
            ffts[ifo] = np.abs(fft.value)

        return freqs, ffts

    def qscan(self, strain: Dict[str, np.ndarray]):
        qscans = []
        for ifo in self.ifos:
            data = strain[ifo]
            ts = TimeSeries(data, times=self.whitened_times)
            ts = ts.crop(-3, 3)
            qscan = ts.q_transform(
                logf=True, frange=(32, 1024), whiten=False, outseg=(-1, 1)
            )
            qscans.append(qscan)
        return qscans
