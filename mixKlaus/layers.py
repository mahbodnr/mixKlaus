from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnmf import NNMFLayer, NonNegativeParameter

from mixKlaus.utils import anderson


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        features: int,
        mlp_hidden: int,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        save_attn_map: bool = False,
    ):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(features)
        self.attention = MultiHeadSelfAttention(
            features, head=head, dropout=dropout, save_attn_map=save_attn_map
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
        head: int = 8,
        dropout: float = 0.0,
        save_attn_map: bool = False,
    ):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.features = features
        self.sqrt_d = self.features**0.5

        self.Wq = nn.Linear(features, features)
        self.Wk = nn.Linear(features, features)
        self.Wv = nn.Linear(features, features)

        self.out_project = nn.Linear(features, features)
        self.dropout = nn.Dropout(dropout)

        self.save_attn_map = save_attn_map

    def forward(self, x):
        B, T, _ = x.size()  # (#Batches, #Inputs, #Features)
        Q = self.Wq(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.head, self.features // self.head).transpose(1, 2)

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
        seq_len: int,
        mlp_hidden: int,
        backward_method: str,
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
    ):
        super(NNMFMixerEncoder, self).__init__(
            features, mlp_hidden, head, dropout, use_mlp
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
                seq_len=seq_len,
                heads=head,
                n_iterations=n_iterations,
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
            )
        else:
            self.attention = NNMFMixerAttentionHeads(
                features=features,
                seq_len=seq_len,
                heads=head,
                n_iterations=n_iterations,
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
            )

    def forward(self, x):
        x = self.la1(x)
        x = torch.clamp(x, min=0.0001)
        out = self.attention(x) + x
        if self.mlp is not None:
            out = self.mlp(self.la2(out)) + out
        return out


class NNMFMixerAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
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
        )
        self.threshold: float = 0.00001
        self.heads: int = heads

        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(features // heads, features // heads)
        )
        self.global_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(seq_len, seq_len)
        )

        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)

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
        self.h = F.normalize(torch.ones_like(x), p=1, dim=-1)

    def _reconstruct(self, h):
        h = torch.einsum("bohf,oi->bihf", h, self.global_weight)
        return F.linear(h, self.local_weight.t())

    def _forward(self, x):
        x = F.linear(x, self.local_weight)
        return torch.einsum("bihf,oi->bohf", x, self.global_weight)

    def _process_h(self, h):
        h = self._secure_tensor(h)
        if self.normalize_reconstruction:
            h = F.normalize(h, p=1, dim=self.normalize_reconstruction_dim)
        # TODO: apply sparsity
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        x = super().forward(x)
        x = x.flatten(-2)
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
        stride: int,
        padding: int,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
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
        )
        self.threshold: float = 0.00001
        self.heads: int = heads
        self.features: int = features
        self.seq_len: int = seq_len

        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(features // heads, features // heads)
        )

        self.patch_size = int(self.seq_len**0.5)
        assert (
            self.patch_size**2 == self.seq_len
        ), "seq_len does not matches the patch size. Check if you are using an <CLS> token."
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        assert (
            kernel_size == -(seq_len - 1) * (stride - 1) + 2 * padding + 1
        ), "Provided kernel size, stride and padding does not apply a 'same padding' convolution to the input with 'seq_len'."
        self.global_weight = nn.Parameter(
            torch.rand(features, self.heads, kernel_size, kernel_size)
        )

        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)

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

    def _make_global_weight(self):
        return F.normalize(
            self.global_weight.repeat_interleave(self.features // self.heads, dim=1),
            p=1,
            dim=(1, 2, 3),
        )  # output_channels, input_channels, kernel_size, kernel_size

    def _reconstruct(self, h):
        # h: B, T, F (=H*D)
        h = h.reshape(h.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        h = F.conv_transpose2d(
            h, self.global_weight_conv, stride=self.stride, padding=self.padding
        )
        h = h.flatten(-2).permute(0, 2, 1)  # B, T, HD
        h = h.reshape(h.shape[0], h.shape[1], self.heads, -1)  # B, T, H, D
        return F.linear(h, self.local_weight.t()).flatten(-2)  # B, T, F

    def _forward(self, x):
        # x: B, T, F (=H*D)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        x = F.linear(x, self.local_weight)
        x = x.reshape(x.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        x = F.conv2d(
            x, self.global_weight_conv, stride=self.stride, padding=self.padding
        )
        x = x.flatten(-2).permute(0, 2, 1)  # B, T, HD
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_weight_conv = self._make_global_weight()
        x = super().forward(x)
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def _reset_h(self, x):
        self.h = F.normalize(torch.ones_like(x), p=1, dim=-1)

    def _check_forward(self, input):
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()
        assert (self.local_weight >= 0).all(), self.local_weight.min()

    def _process_h(self, h):
        # raise NotImplementedError
        h = self._secure_tensor(h)
        if self.normalize_reconstruction:
            h = F.normalize(h, p=1, dim=self.normalize_reconstruction_dim)
        # TODO: apply sparsity
        return h


class BaselineMixerAttentionHeads(nn.Module):
    def __init__(self, features, seq_len, heads):
        super().__init__()
        assert features % heads == 0
        self.heads = heads
        self.features = features
        self.seq_len = seq_len
        self.local_mlp = nn.Linear(features // heads, features // heads)
        self.global_weight = nn.Parameter(
            torch.rand(seq_len, seq_len), requires_grad=True
        )
        self.out_project = nn.Linear(features, features)

    def forward(self, x):
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
        ffn_features: int,
        heads: int,
        mlp_hidden: int,
        dropout: float = 0.0,
        use_mlp: bool = True,
    ):
        super(BaselineMixerEncoder, self).__init__(
            features, mlp_hidden, heads, dropout, use_mlp
        )
        self.attention = BaselineMixerAttentionHeads(
            features,
            seq_len,
            heads,
        )
