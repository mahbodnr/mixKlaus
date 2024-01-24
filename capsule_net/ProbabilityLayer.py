import torch


class ProbabilityLayer(torch.nn.Module):
    probability: None | torch.Tensor = None

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return  input.pow(2).sum(dim=2).sqrt()
