import torch


class MarginLoss(torch.nn.Module):
    m_pos: float
    m_neg: float
    lambda_: float

    def __init__(self, m_pos: float, m_neg: float, lambda_: float) -> None:
        super().__init__()
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.lambda_ = lambda_

    def forward(
        self, lengths: torch.Tensor, targets: torch.Tensor, size_average: bool = True
    ) -> torch.Tensor:
        t = torch.zeros_like(lengths, dtype=torch.int64, device=targets.device)

        targets = t.scatter_(1, targets.unsqueeze(-1), 1).type(
            torch.get_default_dtype()
        )

        losses = targets * torch.nn.functional.relu(self.m_pos - lengths) ** 2

        losses = (
            losses
            + self.lambda_
            * (1.0 - targets)
            * torch.nn.functional.relu(lengths - self.m_neg) ** 2
        )

        return losses.mean() if size_average else losses.sum()
