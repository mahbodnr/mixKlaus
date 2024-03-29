import torch
import torch.nn as nn

from mixKlaus.layers import (
    TransformerEncoder,
    BaselineMixerEncoder,
    NNMFMixerEncoder,
    AlphaMixerEncoder,
)


class ViT(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        dropout: float = 0.0,
        num_layers: int = 12,
        hidden: int = 768,
        embed_dim: int = 768,
        encoder_mlp: bool = True,
        mlp_hidden: int = 768 * 4,
        head: int = 8,
        is_cls_token: bool = True,
    ):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch  # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size // self.patch
        assert (
            self.patch_size * self.patch == img_size
        ), f"img_size must be divisible by patch. Got {img_size} and {patch}"
        f = (img_size // self.patch) ** 2 * in_c  # 48 # patch vec length
        num_tokens = (self.patch**2) + 1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden)  # (b, n, f)
        self.cls_token = (
            nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        )
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden))
        self.enc = nn.Sequential(
            *[
                TransformerEncoder(
                    features=hidden,
                    embed_dim=embed_dim,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    head=head,
                    use_mlp=encoder_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, num_classes)  # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = (
            x.unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
            .permute(0, 2, 3, 4, 5, 1)
        )
        out = out.reshape(x.size(0), self.patch**2, -1)
        return out


class NNMFMixer(ViT):
    def __init__(
        self,
        conv: bool,
        dynamic_weight: bool,
        kernel_size: int,
        stride: int,
        padding: int,
        nnmf_iterations: int,
        nnmf_backward: str,
        nnmf_output: str,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        dropout: float = 0.0,
        num_layers: int = 12,
        hidden: int = 768,
        embed_dim: int = 768,
        nnmf_hidden: int | None = None,
        nnmf_seq_len: int | None = None,
        alpha_dynamics_iterations: int = 0,
        encoder_mlp: bool = True,
        nnmf_skip_connection: bool = True,
        mlp_hidden: int = 768 * 4,
        head: int = 8,
        gated: bool = False,
        is_cls_token: bool = True,
        pos_emb: bool = True,
        output_mode: str = "mean",
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = None,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = None,
        normalize_hidden: bool = True,
        normalize_hidden_dim: int | None = None,
        h_softmax_power: float = 1.0,
        convergence_threshold: float = 0.0,
    ):
        super(NNMFMixer, self).__init__(
            in_c,
            num_classes,
            img_size,
            patch,
            dropout,
            num_layers,
            hidden,
            embed_dim,
            encoder_mlp,
            mlp_hidden,
            head,
            is_cls_token,
        )
        self.seq_len = self.patch**2 + 1 if self.is_cls_token else self.patch**2
        self.enc = nn.Sequential(
            *[
                NNMFMixerEncoder(
                    n_iterations=nnmf_iterations,
                    features=hidden,
                    embed_dim=embed_dim,
                    seq_len=self.seq_len,
                    hidden_features=nnmf_hidden,
                    hidden_seq_len=nnmf_seq_len,
                    mlp_hidden=mlp_hidden,
                    head=head,
                    gated=gated,
                    skip_connection=nnmf_skip_connection,
                    dropout=dropout,
                    use_mlp=encoder_mlp,
                    output=nnmf_output,
                    backward_method=nnmf_backward,
                    conv=conv,
                    alpha_dynamics_iterations=alpha_dynamics_iterations,
                    dynamic_weight=dynamic_weight,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    normalize_input=normalize_input,
                    divide_input=divide_input,
                    normalize_input_dim=normalize_input_dim,
                    normalize_reconstruction=normalize_reconstruction,
                    normalize_reconstruction_dim=normalize_reconstruction_dim,
                    normalize_h=normalize_hidden,
                    normalize_h_dim=normalize_hidden_dim,
                    h_softmax_power=h_softmax_power,
                    convergence_threshold=convergence_threshold,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_mode = output_mode
        if output_mode == "fc":
            self.out_fc = nn.Linear(hidden * self.seq_len, hidden)
        elif output_mode == "mixer":
            self.out_mixer = NNMFMixerEncoder(
                n_iterations=nnmf_iterations,
                features=hidden,
                embed_dim=embed_dim,
                seq_len=self.seq_len,
                hidden_features=nnmf_hidden,
                hidden_seq_len=1,
                mlp_hidden=mlp_hidden,
                head=head,
                gated=gated,
                skip_connection=False,
                dropout=dropout,
                use_mlp=encoder_mlp,
                output="hidden",
                backward_method=nnmf_backward,
                alpha_dynamics_iterations=alpha_dynamics_iterations,
                conv=False,
                dynamic_weight=dynamic_weight,
                normalize_input=normalize_input,
                divide_input=divide_input,
                normalize_input_dim=normalize_input_dim,
                normalize_reconstruction=normalize_reconstruction,
                normalize_reconstruction_dim=normalize_reconstruction_dim,
                normalize_h=normalize_hidden,
                normalize_h_dim=normalize_hidden_dim,
                h_softmax_power=h_softmax_power,
                convergence_threshold=convergence_threshold,
            )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            if self.output_mode == "mean":
                out = out.mean(1)
            elif self.output_mode == "fc":
                out = self.out_fc(out.flatten(1))
            elif self.output_mode == "mixer":
                out = self.out_mixer(out).squeeze(1)
            else:
                raise NotImplementedError
        out = self.fc(out)
        return out


class AlphaMixer(NNMFMixer):
    def __init__(
        self,
        conv: bool,
        dynamic_weight: bool,
        kernel_size: int,
        stride: int,
        padding: int,
        nnmf_iterations: int,
        nnmf_backward: str,
        nnmf_output: str,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        dropout: float = 0.0,
        num_layers: int = 12,
        hidden: int = 768,
        embed_dim: int = 768,
        nnmf_hidden: int | None = None,
        nnmf_seq_len: int | None = None,
        alpha_dynamics_iterations: int = 0,
        encoder_mlp: bool = True,
        nnmf_skip_connection: bool = True,
        mlp_hidden: int = 768 * 4,
        head: int = 8,
        gated: bool = False,
        is_cls_token: bool = True,
        pos_emb: bool = True,
        output_mode: str = "mean",
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = None,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = None,
        normalize_hidden: bool = True,
        normalize_hidden_dim: int | None = None,
        h_softmax_power: float = 1.0,
        convergence_threshold: float = 0.0,
    ):
        super(AlphaMixer, self).__init__(
            conv,
            dynamic_weight,
            kernel_size,
            stride,
            padding,
            nnmf_iterations,
            nnmf_backward,
            nnmf_output,
            in_c,
            num_classes,
            img_size,
            patch,
            dropout,
            num_layers,
            hidden,
            embed_dim,
            nnmf_hidden,
            nnmf_seq_len,
            alpha_dynamics_iterations,
            encoder_mlp,
            nnmf_skip_connection,
            mlp_hidden,
            head,
            gated,
            is_cls_token,
            pos_emb,
            output_mode,
            normalize_input,
            divide_input,
            normalize_input_dim,
            normalize_reconstruction,
            normalize_reconstruction_dim,
            normalize_hidden,
            normalize_hidden_dim,
            h_softmax_power,
            convergence_threshold,
        )
        if conv:
            raise NotImplementedError("AlphaMixer does not support convolutions")
        if dynamic_weight:
            raise NotImplementedError("AlphaMixer does not support dynamic weights")
        if nnmf_output != "hidden":
            raise NotImplementedError("AlphaMixer only supports hidden output")
        
        self.enc = nn.Sequential(
            *[
                AlphaMixerEncoder(
                    n_iterations=nnmf_iterations,
                    alpha_dynamics_iterations=alpha_dynamics_iterations,
                    features=hidden,
                    embed_dim=embed_dim,
                    seq_len=self.seq_len,
                    hidden_features=nnmf_hidden,
                    hidden_seq_len=nnmf_seq_len,
                    mlp_hidden=mlp_hidden,
                    head=head,
                    gated=gated,
                    skip_connection=nnmf_skip_connection,
                    dropout=dropout,
                    use_mlp=encoder_mlp,
                    output=nnmf_output,
                    backward_method=nnmf_backward,
                    convergence_threshold=convergence_threshold,
                    normalize_input=normalize_input,
                    divide_input=divide_input,
                    normalize_input_dim=normalize_input_dim,
                    normalize_reconstruction=normalize_reconstruction,
                    normalize_reconstruction_dim=normalize_reconstruction_dim,
                    normalize_h=normalize_hidden,
                    normalize_h_dim=normalize_hidden_dim,
                    h_softmax_power=h_softmax_power,
                )
                for _ in range(num_layers)
            ]
        )
        if output_mode=="mixer":
            self.out_mixer = AlphaMixerEncoder(
                n_iterations=nnmf_iterations,
                alpha_dynamics_iterations=alpha_dynamics_iterations,
                features=hidden,
                embed_dim=embed_dim,
                seq_len=self.seq_len,
                hidden_features=nnmf_hidden,
                hidden_seq_len=1,
                mlp_hidden=mlp_hidden,
                head=head,
                gated=gated,
                skip_connection=False,
                dropout=dropout,
                use_mlp=encoder_mlp,
                output="hidden",
                backward_method=nnmf_backward,
                convergence_threshold=convergence_threshold,
                normalize_input=normalize_input,
                divide_input=divide_input,
                normalize_input_dim=normalize_input_dim,
                normalize_reconstruction=normalize_reconstruction,
                normalize_reconstruction_dim=normalize_reconstruction_dim,
                normalize_h=normalize_hidden,
                normalize_h_dim=normalize_hidden_dim,
                h_softmax_power=h_softmax_power,
            )

class BaselineMixer(ViT):
    def __init__(
        self,
        seq_len: int,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 224,
        patch: int = 16,
        num_layers: int = 12,
        hidden: int = 768,
        embed_dim: int = 768,
        encoder_mlp: bool = True,
        mlp_hidden: int = 768 * 4,
        dropout: float = 0.0,
        head: int = 8,
        is_cls_token: bool = True,
        pos_emb: bool = True,
    ):
        super(BaselineMixer, self).__init__(
            in_c,
            num_classes,
            img_size,
            patch,
            dropout,
            num_layers,
            hidden,
            embed_dim,
            encoder_mlp,
            mlp_hidden,
            head,
            is_cls_token,
        )
        self.enc = nn.Sequential(
            *[
                BaselineMixerEncoder(
                    seq_len=seq_len,
                    features=hidden,
                    embed_dim=embed_dim,
                    ffn_features=mlp_hidden,
                    heads=head,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    use_mlp=encoder_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        if not pos_emb:
            self.pos_emb = torch.zeros_like(self.pos_emb, requires_grad=False)
