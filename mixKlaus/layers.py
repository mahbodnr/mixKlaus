import torch
import torch.nn as nn
import torch.nn.functional as F

from nnmf import NNMFLayer, NonNegativeParameter

from mixKlaus.utils import anderson

MINIMUM_POSITIVE = 1e-6


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        features: int,
        embed_dim: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        save_attn_map: bool = False,
    ):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(features)
        self.attention = MultiHeadSelfAttention(
            features, embed_dim, head=head, dropout=dropout, save_attn_map=save_attn_map
        )
        self.la2 = nn.LayerNorm(features)
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(features, mlp_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, features),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = None
        self._save_attn_map = save_attn_map

    def forward(self, x):
        out = self.attention(self.la1(x)) + x
        # out = self.attention(x) + x
        if self.mlp is not None:
            out = self.mlp(self.la2(out)) + out
        return out

    @property
    def save_attn_map(self):
        return self._save_attn_map

    @save_attn_map.setter
    def save_attn_map(self, value):
        self._save_attn_map = value
        self.attention.save_attn_map = value

    def get_attention_map(self):
        if self._save_attn_map:
            return self.attention.attn_map
        else:
            raise Exception(
                "Attention map was not saved. Set save_attn_map=True when initializing the model."
            )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        features: int,
        embed_dim: int,
        head: int = 8,
        dropout: float = 0.0,
        save_attn_map: bool = False,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.features = features
        self.embed_dim = embed_dim
        assert (
            self.embed_dim % self.head == 0
        ), f"Incompatible features: {self.embed_dim} % {self.head} != 0"
        self.sqrt_d = self.features**0.5

        self.Wq = nn.Linear(features, embed_dim)
        self.Wk = nn.Linear(features, embed_dim)
        self.Wv = nn.Linear(features, embed_dim)

        self.out_project = nn.Linear(embed_dim, features)
        self.dropout = nn.Dropout(dropout)

        self.save_attn_map = save_attn_map

    def forward(self, x):
        B, T, _ = x.size()  # (#Batches, #Inputs, #Features)
        Q = (
            self.Wq(x)
            .view(B, T, self.head, self.embed_dim // self.head)
            .transpose(1, 2)
        )
        K = (
            self.Wk(x)
            .view(B, T, self.head, self.embed_dim // self.head)
            .transpose(1, 2)
        )
        V = (
            self.Wv(x)
            .view(B, T, self.head, self.embed_dim // self.head)
            .transpose(1, 2)
        )

        attn_map = F.softmax(
            torch.einsum("bhif, bhjf->bhij", Q, K) / self.sqrt_d, dim=-1
        )  # (b,h,n,n)
        if self.save_attn_map:
            self.attn_map = attn_map
        attn = torch.einsum("bhij, bhjf->bihf", attn_map, V)  # (b,n,h,f//h)
        output = self.dropout(self.out_project(attn.flatten(2)))
        return output


class NNMFMixerEncoder(TransformerEncoder):
    def __init__(
        self,
        n_iterations: int,
        features: int,
        embed_dim: int,
        seq_len: int,
        mlp_hidden: int,
        output: str,
        backward_method: str,
        hidden_features: int | None = None,
        hidden_seq_len: int | None = None,
        gated: bool = False,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        use_out_proj: bool = True,
        conv: bool = False,
        kernel_size: int | None = None,
        stride: int | None = None,
        padding: int | None = None,
        normalize_input: bool = True,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
    ):
        super(NNMFMixerEncoder, self).__init__(
            features,
            embed_dim,
            mlp_hidden,
            head=head,
            dropout=dropout,
            use_mlp=use_mlp,
            save_attn_map=False,
        )
        if conv:
            assert kernel_size is not None
            assert stride is not None
            assert padding is not None
            self.attention = NNMFMixerAttentionHeadsConv(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                features=features,
                embed_dim=embed_dim,
                seq_len=seq_len,
                heads=head,
                hidden_features=hidden_features,
                n_iterations=n_iterations,
                gated=gated,
                output=output,
                backward_method=backward_method,
                h_update_rate=1,
                keep_h=False,
                activate_secure_tensors=True,
                use_out_proj=use_out_proj,
                solver=anderson,
                normalize_input=normalize_input,
                normalize_input_dim=normalize_input_dim,
                normalize_reconstruction=normalize_reconstruction,
                normalize_reconstruction_dim=normalize_reconstruction_dim,
                normalize_h=normalize_h,
                normalize_h_dim=normalize_h_dim,
            )
        else:
            self.attention = NNMFMixerAttentionHeads(
                features=features,
                embed_dim=embed_dim,
                seq_len=seq_len,
                hidden_features=hidden_features,
                hidden_seq_len=hidden_seq_len,
                heads=head,
                n_iterations=n_iterations,
                gated=gated,
                output=output,
                backward_method=backward_method,
                h_update_rate=1,
                keep_h=False,
                activate_secure_tensors=True,
                use_out_proj=use_out_proj,
                solver=anderson,
                normalize_input=normalize_input,
                normalize_input_dim=normalize_input_dim,
                normalize_reconstruction=normalize_reconstruction,
                normalize_reconstruction_dim=normalize_reconstruction_dim,
                normalize_h=normalize_h,
                normalize_h_dim=normalize_h_dim,
            )


class NNMFMixerAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        embed_dim: int,
        heads: int,
        n_iterations: int,
        output: str,
        hidden_features: int | None = None,
        hidden_seq_len: int | None = None,
        gated: bool = False,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        solver=None,
        normalize_input=True,
        normalize_input_dim=-1,
        normalize_reconstruction=True,
        normalize_reconstruction_dim=-1,
        normalize_h=True,
        normalize_h_dim=-1,
    ):
        super().__init__(
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            solver=solver,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            return_reconstruction=True,
        )
        self.threshold: float = 0.00001
        self.heads: int = heads
        assert output in ["reconstruction", "hidden"]
        self.output: str = output
        self.features: int = features
        self.seq_len: int = seq_len
        self.embed_dim: int = embed_dim
        self.hidden_features: int = (
            embed_dim if hidden_features is None else hidden_features
        )
        assert self.hidden_features > heads and (
            self.hidden_features % heads == 0
        ), f"Incompatible hidden features: {self.hidden_features}, having heads: {heads}"
        self.hidden_seq_len: int = seq_len if hidden_seq_len is None else hidden_seq_len
        self.normalize_h = normalize_h
        self.normalize_h_dim = normalize_h_dim

        self.embed = nn.Linear(features, embed_dim)

        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.hidden_features // heads, embed_dim // heads)
        )
        self.global_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.hidden_seq_len, seq_len)
        )

        self.gated = gated
        if self.gated:
            self.gate = nn.Linear(features, embed_dim)
            self.gate_activation = nn.SiLU()

        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(embed_dim, features)

        self.save_attn_map = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        for weight in [self.local_weight, self.global_weight]:
            weight_data = F.normalize(
                weight.data, p=1, dim=1
            )  # May contain negative values if Madam not used
            torch.clamp(
                weight_data,
                min=self.threshold,
                max=None,
                out=weight.data,
            )
            weight.data = F.normalize(weight.data, p=1, dim=1)

    def _reset_h(self, x):
        self.h = torch.rand(
            (
                x.shape[0],
                self.hidden_seq_len,
                self.heads,
                self.hidden_features // self.heads,
            ),
            device=x.device,
        )
        if self.normalize_h:
            self.h = F.normalize(self.h, p=1, dim=self.normalize_h_dim)

    def _reconstruct(self, h):
        h = torch.einsum("bohf,oi->bihf", h, self.global_weight)
        return F.linear(h, self.local_weight.t())

    def _forward(self, x):
        x = F.linear(x, self.local_weight)
        return torch.einsum("bihf,oi->bohf", x, self.global_weight)

    def _process_h(self, h):
        h = self._secure_tensor(h)
        if self.normalize_h:
            h = F.normalize(h, p=1, dim=self.normalize_h_dim)
        # TODO: apply sparsity
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            z = self.gate_activation(self.gate(x))
        x = self.embed(x)
        x = torch.clamp(x, min=MINIMUM_POSITIVE)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        out = {}
        out["hidden"], out["reconstruction"] = super().forward(x)
        x = out[self.output]
        x = x.flatten(-2)
        if self.gated:
            x = x * z
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def _check_forward(self, input):
        assert (self.local_weight >= 0).all(), self.local_weight.min()
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()


class NNMFMixerAttentionHeadsConv(NNMFLayer):
    def __init__(
        self,
        kernel_size: int,
        embed_dim: int,
        stride: int,
        padding: int,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
        output: str,
        hidden_features: int | None = None,
        gated: bool = False,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        solver: callable = None,
        normalize_input: bool = True,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
    ) -> None:
        super().__init__(
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            solver=solver,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            return_reconstruction=True,
        )
        self.threshold: float = 0.00001
        self.heads: int = heads
        self.features: int = features
        self.seq_len: int = seq_len
        self.embed_dim: int = embed_dim
        self.hidden_features: int = (
            embed_dim if hidden_features is None else hidden_features
        )
        assert output in ["reconstruction", "hidden"]
        self.output: str = output
        self.normalize_h = normalize_h
        self.normalize_h_dim = normalize_h_dim

        assert self.hidden_features > heads and (
            self.hidden_features % heads == 0
        ), f"Incompatible hidden features: {self.hidden_features}, having heads: {heads}"
        self.embed = nn.Linear(features, embed_dim)

        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.hidden_features // heads, embed_dim // heads)
        )

        self.patch_size = int(self.seq_len**0.5)
        assert (
            self.patch_size**2 == self.seq_len
        ), "seq_len does not matches the patch size. Check if you are using an <CLS> token."
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_size = self.get_output_size(
            self.patch_size, self.patch_size, kernel_size, stride, padding
        )
        if (
            self.output_size != (self.patch_size, self.patch_size)
            and self.output == "hidden"
        ):
            print(
                f"[Warning] Provided kernel size, stride and padding does not apply a 'same padding' convolution to the input with 'seq_len'."
            )
        self.hidden_seq_len = self.output_size[0] * self.output_size[1]
        self.global_weight = nn.Parameter(
            torch.rand(self.hidden_features, self.heads, kernel_size, kernel_size)
        )

        self.gated = gated
        if self.gated:
            self.gate = nn.Linear(features, embed_dim)
            self.gate_activation = nn.SiLU()

        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(embed_dim, features)

        self.save_attn_map = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        weight_data = F.normalize(
            self.local_weight.data, p=1, dim=1
        )  # May contain negative values if Madam not used
        torch.clamp(
            weight_data,
            min=self.threshold,
            max=None,
            out=self.local_weight.data,
        )
        self.local_weight.data = F.normalize(self.local_weight.data, p=1, dim=1)

        torch.clamp(
            self.global_weight.data,
            min=self.threshold,
            max=None,
            out=self.global_weight.data,
        )

    def _make_global_weight(self) -> torch.Tensor:
        return F.normalize(
            self.global_weight.repeat_interleave(self.hidden_features // self.heads, dim=1),
            p=1,
            dim=(1, 2, 3),
        )  # output_channels, input_channels, kernel_size, kernel_size

    def _reconstruct(self, h: torch.Tensor) -> torch.Tensor:
        # h: B, T, H, D
        h = h.reshape(h.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        h = F.conv_transpose2d(
            h, self.global_weight_conv, stride=self.stride, padding=self.padding
        )  # B, HD, P, P
        h = h.flatten(-2).permute(0, 2, 1)  # B, T, HD
        h = h.reshape(h.shape[0], h.shape[1], self.heads, -1)  # B, T, H, D
        return F.linear(h, self.local_weight.t())  # B, T, H, D

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, T, H, D
        x = F.linear(x, self.local_weight)  # B, T, H, D
        x = x.reshape(x.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        x = F.conv2d(
            x, self.global_weight_conv, stride=self.stride, padding=self.padding
        )  # B, HD, P, P
        x = x.flatten(-2).permute(0, 2, 1)  # B, T, HD
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            z = self.gate_activation(self.gate(x))
        x = self.embed(x)
        self.global_weight_conv = self._make_global_weight()
        x = torch.clamp(x, min=MINIMUM_POSITIVE)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        out = {}
        out["hidden"], out["reconstruction"] = super().forward(x)
        x = out[self.output]
        x = x.flatten(-2)
        if self.gated:
            x = x * z
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def _reset_h(self, x):
        self.h = torch.rand(
            (
                x.shape[0],
                self.hidden_seq_len,
                self.heads,
                self.hidden_features // self.heads,
            ),
            device=x.device,
        )
        if self.normalize_h:
            self.h = F.normalize(self.h, p=1, dim=self.normalize_h_dim)

    def _check_forward(self, input):
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()
        assert (self.local_weight >= 0).all(), self.local_weight.min()

    def _process_h(self, h):
        h = self._secure_tensor(h)
        if self.normalize_h:
            h = F.normalize(h, p=1, dim=self.normalize_h_dim)
        # TODO: apply sparsity
        return h

    @staticmethod
    def get_output_size(Hin, Win, kernel_size, stride, padding):
        Hout = (Hin - kernel_size + 2 * padding) // stride + 1
        Wout = (Win - kernel_size + 2 * padding) // stride + 1
        return Hout, Wout


class BaselineMixerAttentionHeads(nn.Module):
    def __init__(self, features, embed_dim, seq_len, heads):
        super().__init__()
        assert features % heads == 0
        self.heads = heads
        self.features = features
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.embed = nn.Linear(features, embed_dim)
        self.local_mlp = nn.Linear(features // heads, features // heads)
        self.global_weight = nn.Parameter(
            torch.rand(seq_len, seq_len), requires_grad=True
        )
        self.out_project = nn.Linear(features, features)

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)
        x = self.local_mlp(x)
        x = torch.einsum("bihf,oi->bohf", x, self.global_weight)
        x = self.out_project(x.flatten(2))
        return x


class BaselineMixerEncoder(TransformerEncoder):
    def __init__(
        self,
        seq_len: int,
        features: int,
        embed_dim: int,
        heads: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(BaselineMixerEncoder, self).__init__(
            features, embed_dim, mlp_hidden, heads, dropout, use_mlp
        )
        self.attention = BaselineMixerAttentionHeads(
            features,
            seq_len,
            heads,
        )
