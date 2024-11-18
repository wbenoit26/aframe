from typing import TYPE_CHECKING, List, Optional

import torch
from hermes.quiver import Platform
from hermes.quiver.streaming import utils as streaming_utils
from ml4gw.transforms import SingleQTransform
from torchaudio.transforms import Resample

from utils.preprocessing import BackgroundSnapshotter, BatchWhitener

if TYPE_CHECKING:
    from hermes.quiver.model import EnsembleModel, ExposedTensor


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


def scale_model(model, instances):
    """
    Scale the model to the number of instances per GPU desired
    at inference time
    """
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


def add_streaming_input_preprocessor(
    ensemble: "EnsembleModel",
    input: "ExposedTensor",
    psd_length: float,
    sample_rate: float,
    kernel_length: float,
    inference_sampling_rate: float,
    fduration: float,
    fftlength: float,
    q: Optional[float] = None,
    highpass: Optional[float] = None,
    preproc_instances: Optional[int] = None,
    streams_per_gpu: int = 1,
) -> "ExposedTensor":
    """Create a snapshotter model and add it to the repository"""

    batch_size, num_ifos, *kernel_size = input.shape
    if q is not None:
        if len(kernel_size) != 2:
            raise ValueError(
                "If q is not None, the input kernel should be 2D, "
                f"got {len(kernel_size)} dimension(s)"
            )
        augmentor = SingleQTransform(
            duration=kernel_length,
            sample_rate=sample_rate,
            spectrogram_shape=kernel_size,
            q=q,
        )
    else:
        augmentor = None
    augmentor = MultiRateResample(
        original_sample_rate=sample_rate,
        duration=kernel_length,
        new_sample_rates=[128, 256, 512, 1024, 2048],
        breakpoints=[4, 5, 5.5, 5.75],
    )

    snapshotter = BackgroundSnapshotter(
        psd_length=psd_length,
        kernel_length=kernel_length,
        fduration=fduration,
        sample_rate=sample_rate,
        inference_sampling_rate=inference_sampling_rate,
    )

    stride = int(sample_rate / inference_sampling_rate)
    state_shape = (2, num_ifos, snapshotter.state_size)
    input_shape = (2, num_ifos, batch_size * stride)
    streaming_model = streaming_utils.add_streaming_model(
        ensemble.repository,
        streaming_layer=snapshotter,
        name="snapshotter",
        input_name="stream",
        input_shape=input_shape,
        state_names=["snapshot"],
        state_shapes=[state_shape],
        output_names=["strain"],
        streams_per_gpu=streams_per_gpu,
    )
    ensemble.add_input(streaming_model.inputs["stream"])

    preprocessor = BatchWhitener(
        kernel_length=kernel_length,
        sample_rate=sample_rate,
        batch_size=batch_size,
        inference_sampling_rate=inference_sampling_rate,
        fduration=fduration,
        fftlength=fftlength,
        highpass=highpass,
        augmentor=augmentor,
    )
    preproc_model = ensemble.repository.add(
        "preprocessor", platform=Platform.TORCHSCRIPT
    )
    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if preproc_instances is not None:
        scale_model(preproc_model, preproc_instances)

    input_shape = streaming_model.outputs["strain"].shape
    preproc_model.export_version(
        preprocessor,
        input_shapes={"strain": input_shape},
        output_names=["whitened"],
    )
    ensemble.pipe(
        streaming_model.outputs["strain"],
        preproc_model.inputs["strain"],
    )
    return preproc_model.outputs["whitened"]
