import torch
import torch.nn.functional as F

from .base import BaseGRFKernel


def diffusion_modulation(length: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute diffusion modulator vector: (-beta)^length / (2^length * Gamma(length + 1))
    """
    length = length.to(dtype=beta.dtype, device=beta.device)

    numerator = torch.pow(-beta, length)
    denominator = torch.pow(
        torch.tensor(2.0, dtype=beta.dtype, device=beta.device), length
    )
    denominator = denominator * torch.exp(torch.lgamma(length + 1.0))

    return numerator / denominator


class GRFDiffusionKernel(BaseGRFKernel):
    """
    GRF kernel with diffusion modulation.
    """

    def __init__(self, max_walk_length, step_matrices_torch, **kwargs):
        super().__init__(step_matrices_torch=step_matrices_torch, **kwargs)

        self.max_walk_length = max_walk_length
        self.register_parameter(
            name="raw_beta",
            parameter=torch.nn.Parameter(torch.tensor(1.0)),
        )

        self.register_parameter(
            name="raw_sigma_f",
            parameter=torch.nn.Parameter(torch.tensor(1.0)),
        )

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta)

    @property
    def sigma_f(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_f)

    @property
    def modulation_function(self) -> torch.Tensor:
        walk_lengths = torch.arange(
            self.max_walk_length, dtype=self.raw_beta.dtype, device=self.raw_beta.device
        )
        return self.sigma_f * diffusion_modulation(walk_lengths, self.beta)
