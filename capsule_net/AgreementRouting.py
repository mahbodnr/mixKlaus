import torch


class AgreementRouting(torch.nn.Module):
    n_iterations: int
    input_caps: int
    output_caps: int

    def __init__(self, input_caps: int, output_caps: int, n_iterations: int) -> None:
        super().__init__()
        assert n_iterations >= 0
        self.n_iterations: int = n_iterations
        self.input_caps: int = input_caps
        self.output_caps: int = output_caps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b: torch.Tensor = torch.zeros(
            (self.input_caps, self.output_caps), device=input.device, dtype=input.dtype
        )
        
        s = (torch.nn.functional.softmax(b, dim=-1).unsqueeze(2) * input).sum(
            dim=1
        )
        lengths2 = (s**2).sum(dim=2)
        v = s * (lengths2.sqrt() / (1 + lengths2)).unsqueeze(-1)

        b_batch = b.unsqueeze(0)
        for _ in range(0, self.n_iterations):
            b_batch = b_batch + (input * v.unsqueeze(1)).sum(dim=-1)

            s = (
                torch.nn.functional.softmax(
                    b_batch,
                    dim=-1,
                ).unsqueeze(-1)
                * input
            ).sum(dim=1)

            lengths2 = (s**2).sum(dim=2)
            v = s * (lengths2.sqrt() / (1 + lengths2)).unsqueeze(-1)

        return v
