from typing import Literal, Optional

import torch
from architectures.supervised import SupervisedArchitecture
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D


class MultiModalPsd(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.

    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        freq_classes: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.classifier = torch.nn.Linear(time_classes + freq_classes, 1)

    def forward(self, X):
        strain, psds = X
        asds = psds**0.5

        asds *= 1e23
        asds = asds.float()

        time_domain_output = self.time_domain_resnet(strain)

        X_fft = torch.fft.rfft(strain)
        num_freqs = X_fft.shape[-1]
        if asds.shape[-1] != num_freqs:
            asds = torch.nn.functional.interpolate(
                asds, size=(num_freqs,), mode="linear"
            )
        inv_asds = 1 / asds
        X_fft = torch.cat((X_fft.real, X_fft.imag, inv_asds), dim=1)
        freq_domain_output = self.freq_psd_resnet(X_fft)

        concat = torch.cat([time_domain_output, freq_domain_output], dim=-1)
        return self.classifier(concat)
