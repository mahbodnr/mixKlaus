import torch


class PrimaryCapsReshapeLayer(torch.nn.Module):
    output_caps: int
    output_dim: int

    def __init__(self, output_caps: int, output_dim: int) -> None:
        super().__init__()
        self.output_caps: int = output_caps
        self.output_dim: int = output_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            input.reshape(
                input.shape[0],
                self.output_caps,
                self.output_dim,
                input.shape[2],
                input.shape[3],
            )
            .movedim(2, -1)
            .flatten(start_dim=1, end_dim=-2)
        )
