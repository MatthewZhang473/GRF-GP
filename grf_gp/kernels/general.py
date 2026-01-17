import torch

from .base import BaseGRFKernel


class GRFGeneralKernel(BaseGRFKernel):
    """
    Learnable GRF kernel with a free modulation vector.
    """

    def __init__(self, max_walk_length, step_matrices_torch, **kwargs):
        super().__init__(step_matrices_torch=step_matrices_torch, **kwargs)
        self.max_walk_length = max_walk_length
        self.register_parameter(
            name="raw_modulation_function",
            parameter=torch.nn.Parameter(torch.randn(max_walk_length)),
        )

    @property
    def modulation_function(self) -> torch.Tensor:
        return self.raw_modulation_function
