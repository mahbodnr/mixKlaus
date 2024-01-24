import torch


class CapsLayer(torch.nn.Module):
    input_dim: int
    input_caps: int
    output_dim: int
    output_caps: int
    weights: torch.nn.Parameter

    def __init__(
        self, input_caps: int, input_dim: int, output_caps: int, output_dim: int
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.input_caps: int = input_caps
        self.output_dim: int = output_dim
        self.output_caps: int = output_caps
        self.weights: torch.nn.Parameter = torch.nn.Parameter(
            torch.Tensor(input_caps, input_dim, output_caps * output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv: torch.Tensor = 1.0 / torch.sqrt(torch.tensor(self.input_caps))
        self.weights.data.uniform_(-float(stdv), float(stdv))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (
            input.unsqueeze(2)
            .matmul(self.weights)
            .reshape(
                input.shape[0],
                input.shape[1],
                self.output_caps,
                self.output_dim,
            )
        )
