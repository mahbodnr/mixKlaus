import torch


class MaskLayer(torch.nn.Module):
    target: None | torch.Tensor = None
    enable: bool = True

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.enable:
            assert self.target is not None

            mask: torch.Tensor = torch.zeros(
                (input.shape[0], input.shape[1]),
                device=input.device,
                requires_grad=False,
            )
            mask.scatter_(1, self.target.unsqueeze(-1), 1.0)

            return input * mask.unsqueeze(-1)
        else:
            return input
