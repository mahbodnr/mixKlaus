from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import DEBUG

if DEBUG:
    import debug.functional as F

from nnmf.modules import NNMFLayer, NNMFDense, NonNegativeParameter, NNMFLayerDynamicWeight
from nnmf.utils import PowerSoftmax

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
        convergence_threshold: float = 0,
        hidden_features: int | None = None,
        hidden_seq_len: int | None = None,
        gated: bool = False,
        skip_connection: bool = True,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        use_out_proj: bool = True,
        conv: bool = False,
        alpha_dynamics_iterations: int = 0,
        dynamic_weight: bool = False,
        kernel_size: int | None = None,
        stride: int | None = None,
        padding: int | None = None,
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
        h_softmax_power: float = 1,
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
        if dynamic_weight:
            if conv:
                assert kernel_size is not None
                assert stride is not None
                assert padding is not None
                self.attention = NNMFMixerAttentionHeadsConvDynamicWeights(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    features=features,
                    embed_dim=embed_dim,
                    seq_len=seq_len,
                    heads=head,
                    n_iterations=n_iterations,
                    gated=gated,
                    output=output,
                    hidden_features=hidden_features,
                    backward_method=backward_method,
                    h_update_rate=1,
                    h_softmax_power=h_softmax_power,
                    keep_h=False,
                    activate_secure_tensors=True,
                    solver=anderson,
                    convergence_threshold=convergence_threshold,
                    normalize_input=normalize_input,
                    normalize_input_dim=normalize_input_dim,
                    normalize_reconstruction=normalize_reconstruction,
                    normalize_reconstruction_dim=normalize_reconstruction_dim,
                    normalize_h=normalize_h,
                    normalize_h_dim=normalize_h_dim,
                )
            else:
                self.attention = NNMFMixerAttentionHeadsDynamicWeights(
                    seq_len=seq_len,
                    features=features,
                    embed_dim=embed_dim,
                    heads=head,
                    n_iterations=n_iterations,
                    output=output,
                    hidden_features=hidden_features,
                    hidden_seq_len=hidden_seq_len,
                    gated=gated,
                    use_out_proj=use_out_proj,
                    backward_method=backward_method,
                    h_update_rate=1,
                    h_softmax_power=h_softmax_power,
                    keep_h=False,
                    activate_secure_tensors=True,
                    solver=anderson,
                    convergence_threshold=convergence_threshold,
                    normalize_input=normalize_input,
                    normalize_input_dim=normalize_input_dim,
                    normalize_reconstruction=normalize_reconstruction,
                    normalize_reconstruction_dim=normalize_reconstruction_dim,
                    normalize_h=normalize_h,
                    normalize_h_dim=normalize_h_dim,
                )
        else:
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
                    alpha_dynamics_iterations=alpha_dynamics_iterations,
                    gated=gated,
                    output=output,
                    backward_method=backward_method,
                    h_update_rate=1,
                    h_softmax_power=h_softmax_power,
                    keep_h=False,
                    activate_secure_tensors=True,
                    use_out_proj=use_out_proj,
                    solver=anderson,                    
                    convergence_threshold=convergence_threshold,
                    normalize_input=normalize_input,
                    divide_input=divide_input,
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
                    alpha_dynamics_iterations=alpha_dynamics_iterations,
                    gated=gated,
                    output=output,
                    backward_method=backward_method,
                    h_update_rate=1,
                    h_softmax_power=h_softmax_power,
                    keep_h=False,
                    activate_secure_tensors=True,
                    use_out_proj=use_out_proj,
                    solver=anderson,
                    convergence_threshold=convergence_threshold,
                    normalize_input=normalize_input,
                    divide_input=divide_input,
                    normalize_input_dim=normalize_input_dim,
                    normalize_reconstruction=normalize_reconstruction,
                    normalize_reconstruction_dim=normalize_reconstruction_dim,
                    normalize_h=normalize_h,
                    normalize_h_dim=normalize_h_dim,
                )

        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.attention(x)
        if self.skip_connection:
            out = out + x
        if self.mlp is not None:
            out = self.mlp(self.la2(out)) + out
        return out


class AlphaMixerEncoder(TransformerEncoder):
    def __init__(
        self,
        n_iterations: int,
        alpha_dynamics_iterations: int,
        features: int,
        embed_dim: int,
        seq_len: int,
        mlp_hidden: int,
        output: str,
        backward_method: str,
        convergence_threshold: float = 0,
        hidden_features: int | None = None,
        hidden_seq_len: int | None = None,
        gated: bool = False,
        skip_connection: bool = True,
        head: int = 8,
        dropout: float = 0.0,
        use_mlp: bool = True,
        use_out_proj: bool = True,
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
        h_softmax_power: float = 1,
        ):
        super(AlphaMixerEncoder, self).__init__(
            features,
            embed_dim,
            mlp_hidden,
            head=head,
            dropout=dropout,
            use_mlp=use_mlp,
            save_attn_map=False,
        )
        self.attention = AplhaMixerAttentionHeads(
            seq_len=seq_len,
            features=features,
            embed_dim=embed_dim,
            heads=head,
            n_iterations=n_iterations,
            output=output,
            hidden_features=hidden_features,
            hidden_seq_len=hidden_seq_len,
            gated=gated,
            use_out_proj=use_out_proj,
            backward_method=backward_method,
            h_update_rate=1,
            h_softmax_power=h_softmax_power,
            keep_h=False,
            activate_secure_tensors=True,
            solver=anderson,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            divide_input=divide_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            normalize_h=normalize_h,
            normalize_h_dim=normalize_h_dim,
            alpha_dynamics_iterations=alpha_dynamics_iterations,
        )
        self.skip_connection = skip_connection

    def forward(self, x):
        out = self.attention(x)
        if self.skip_connection:
            out = out + x
        if self.mlp is not None:
            out = self.mlp(self.la2(out)) + out
        return out


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
        backward_method: str = "fixed_point",
        h_update_rate: float = 1,
        h_softmax_power: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        alpha_dynamics_iterations: int = 0,
        solver=None,
        convergence_threshold=0,
        normalize_input=True,
        divide_input=False,
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
            convergence_threshold=convergence_threshold,
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
        self.divide_input = divide_input
        self.normalize_h = normalize_h
        self.normalize_h_dim = normalize_h_dim
        self.power_softmax = PowerSoftmax(h_softmax_power, dim=self.normalize_h_dim)
        self.alpha_dynamics_iterations = alpha_dynamics_iterations

        self.embed = nn.Linear(features, embed_dim)

        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.hidden_features // heads, embed_dim // heads)
        )
        self.global_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.heads, self.hidden_seq_len, seq_len)
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
        for weight in [self.local_weight, self.global_weight]:
            weight_data = F.normalize(
                weight.data, p=1, dim=-1
            )  # May contain negative values if Madam not used
            torch.clamp(
                weight_data,
                min=self.threshold,
                max=None,
                out=weight.data,
            )
            weight.data = F.normalize(weight.data, p=1, dim=-1)

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
        h = torch.einsum("bohf,hoi->bihf", h, self.global_weight)
        return F.linear(h, self.local_weight.t())

    def _forward(self, x):
        x = F.linear(x, self.local_weight)
        return torch.einsum("bihf,hoi->bohf", x, self.global_weight)

    def _process_h(self, h):
        h = self._secure_tensor(h)
        if self.normalize_h:
            # if power==1 then it is a simple normalization
            h = self.power_softmax(h)
        return h

    def _prepare_input(self, input):
        if self.normalize_input:
            input = F.normalize(input, p=1, dim=self.normalize_input_dim, eps=1e-20)
        if self.divide_input:
            input = input / input.shape[1]
        return input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            z = self.gate_activation(self.gate(x))
        x = self.embed(x)
        x = torch.clamp(x, min=MINIMUM_POSITIVE)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        out = {}
        out["hidden"], out["reconstruction"] = super().forward(x)
        if self.alpha_dynamics_iterations > 0:
            out["hidden"] = self.alpha_dynamics(out["hidden"], x)
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

    # @torch.no_grad()
    def alpha_dynamics(self, h, input):
        input = F.normalize(input, p=1, dim=(-1,-2))/input.shape[1]
        alpha = F.normalize(
            torch.ones(h.shape[0], self.hidden_seq_len).to(h.device),
            p=1,
            dim=1,
        )
        h = F.normalize(h, p=1, dim=(-1, -2))
        h_reconstruction = torch.einsum("bohf,oi->boihf", h, self.global_weight)
        h_reconstruction = F.linear(h_reconstruction, self.local_weight.t())
        self.alpha_convergence = []
        for _ in range(self.alpha_dynamics_iterations):
            alpha_reconstruction = self._reconstruct(
                h * alpha.unsqueeze(-1).unsqueeze(-1)
            )
            new_alpha = alpha * (h_reconstruction * (input / alpha_reconstruction).unsqueeze(1)).sum((-1, -2, -3))
            self.alpha_convergence.append(F.mse_loss(alpha, new_alpha))
            alpha = new_alpha

        return alpha.unsqueeze(-1).unsqueeze(-1) * h


class NNMFMixerAttentionHeadsConv(NNMFMixerAttentionHeads):
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
        backward_method: str = "fixed_point",
        h_update_rate: float = 1,
        h_softmax_power: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        alpha_dynamics_iterations: int = 0,
        solver: callable = None,
        convergence_threshold: float = 0,
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            features=features,
            embed_dim=embed_dim,
            heads=heads,
            n_iterations=n_iterations,
            output=output,
            hidden_features=hidden_features,
            gated=gated,
            use_out_proj=use_out_proj,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            h_softmax_power=h_softmax_power,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            alpha_dynamics_iterations=alpha_dynamics_iterations,
            solver=solver,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            divide_input=divide_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            normalize_h=normalize_h,
            normalize_h_dim=normalize_h_dim,
        )
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
        self.global_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(self.hidden_features, self.heads, kernel_size, kernel_size)
        )
        rec_contributer_factor, forw_contributer_factor = self.get_contributer_factors()
        self.register_buffer(
            "rec_contributer_factor",
            rec_contributer_factor,
        )
        self.register_buffer(
            "forw_contributer_factor",
            forw_contributer_factor,
        )
        self.reset_parameters()

    @torch.no_grad()
    def get_contributer_factors(self):
        temp_model = deepcopy(self)
        temp_model.local_weight.data = torch.ones_like(temp_model.local_weight)
        temp_model.global_weight.data = torch.ones_like(temp_model.global_weight)
        temp_model.normalize_weights()
        temp_model.input_device = torch.device("cpu")
        temp_model.global_weight_conv = temp_model._make_global_weight()
        temp_model.rec_contributer_factor, temp_model.forw_contributer_factor = 1, 1
        h = torch.ones(
            (
                1,
                self.hidden_seq_len,
                self.heads,
                self.hidden_features // self.heads,
            )
        )
        if self.normalize_h:
           h = F.normalize(h, p=1, dim=self.normalize_h_dim)
        rec = temp_model._reconstruct(h)
        rec = rec[0].sum((-1,-2), keepdim=True)
        rec_contributer_factor = 1/(rec / rec.max())
        forw = temp_model._forward(h)
        forw = forw[0].sum((-1,-2), keepdim=True)
        forw_contributer_factor = 1/(forw / forw.max())

        return rec_contributer_factor, forw_contributer_factor

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
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
            self.global_weight.repeat_interleave(
                self.hidden_features // self.heads, dim=1
            ),
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
        return F.linear(h, self.local_weight.t()) * self.rec_contributer_factor # B, T, H, D

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
        return x.reshape(x.shape[0], x.shape[1], self.heads, -1) * self.forw_contributer_factor # B, T, H, D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_weight_conv = self._make_global_weight()
        return super().forward(x)

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

    def alpha_dynamics(self, h, input):
        raise NotImplementedError("Alpha dynamics is not implemented for conv layers.")

    @staticmethod
    def get_output_size(Hin, Win, kernel_size, stride, padding):
        Hout = (Hin - kernel_size + 2 * padding) // stride + 1
        Wout = (Win - kernel_size + 2 * padding) // stride + 1
        return Hout, Wout


class NNMFMixerAttentionHeadsDynamicWeights(
    NNMFLayerDynamicWeight, NNMFMixerAttentionHeads
):
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
        backward_method: str = "fixed_point",
        h_update_rate: float = 1,
        h_softmax_power: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        alpha_dynamics_iterations: int = 0,
        solver=None,
        convergence_threshold=0,
        normalize_input=True,
        divide_input=False,
        normalize_input_dim=-1,
        normalize_reconstruction=True,
        normalize_reconstruction_dim=-1,
        normalize_h=True,
        normalize_h_dim=-1,
    ):
        NNMFLayerDynamicWeight.__init__(
            self,
            n_iterations=n_iterations,
        )
        NNMFMixerAttentionHeads.__init__(
            self,
            seq_len=seq_len,
            features=features,
            embed_dim=embed_dim,
            heads=heads,
            n_iterations=n_iterations,
            alpha_dynamics_iterations=alpha_dynamics_iterations,
            output=output,
            hidden_features=hidden_features,
            hidden_seq_len=hidden_seq_len,
            gated=gated,
            use_out_proj=use_out_proj,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            h_softmax_power=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            solver=solver,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            divide_input=divide_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            normalize_h=normalize_h,
            normalize_h_dim=normalize_h_dim,
        )
        del self.global_weight

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_weight = F.normalize(
            torch.ones(self.hidden_seq_len, self.seq_len), p=1, dim=1
        ).to(x.device)
        return super().forward(x)

    def _update_weight(self, h, reconstruction, input):
        nnmf_update = input / reconstruction
        self.global_weight.data *= torch.einsum("bohf,bihf->oi", h, nnmf_update)
        self.global_weight = self._secure_tensor(self.global_weight)
        self.global_weight = F.normalize(self.global_weight, p=1, dim=1)


class NNMFMixerAttentionHeadsConvDynamicWeights(
    NNMFLayerDynamicWeight, NNMFMixerAttentionHeadsConv
):
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
        backward_method: str = "fixed_point",
        h_update_rate: float = 1,
        h_softmax_power: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        alpha_dynamics_iterations: int = 0,
        solver: callable = None,
        convergence_threshold: float = 0,
        normalize_input: bool = True,
        divide_input: bool = False,
        normalize_input_dim: int | None = -1,
        normalize_reconstruction: bool = True,
        normalize_reconstruction_dim: int | None = -1,
        normalize_h: bool = True,
        normalize_h_dim: int | None = -1,
    ):
        NNMFLayerDynamicWeight.__init__(
            self,
            n_iterations=n_iterations,
        )
        NNMFMixerAttentionHeadsConv.__init__(
            self,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            stride=stride,
            padding=padding,
            seq_len=seq_len,
            features=features,
            heads=heads,
            n_iterations=n_iterations,
            alpha_dynamics_iterations=alpha_dynamics_iterations,
            output=output,
            hidden_features=hidden_features,
            gated=gated,
            use_out_proj=use_out_proj,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            h_softmax_power=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            solver=solver,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            divide_input=divide_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
            normalize_h=normalize_h,
            normalize_h_dim=normalize_h_dim,
        )
        del self.global_weight

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
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

    def _make_global_weight(self) -> torch.Tensor:
        global_weight_conv = torch.ones(
            self.hidden_features,
            self.hidden_features,
            self.kernel_size,
            self.kernel_size,
        ).to(self.input_device)
        return F.normalize(global_weight_conv, p=1, dim=(1, 2, 3))

    def _update_weight(self, h, reconstruction, input):
        nnmf_update = input / reconstruction
        # h:
        h = h.reshape(h.shape[0], self.patch_size, self.patch_size, -1).permute(
            3, 0, 1, 2
        )  # B, T, H, D -> HD, B, P, P
        nnmf_update = nnmf_update.reshape(
            nnmf_update.shape[0], self.patch_size, self.patch_size, -1
        ).permute(
            3, 0, 1, 2
        )  # B, T, H, D -> HD, B, P, P

        # Devide by the batch size to avoid overflow
        nnmf_update /= nnmf_update.shape[1]
        h /= h.shape[1]

        new_weight = F.conv2d(
            h,
            nnmf_update,
            stride=self.stride,
            padding=self.padding,
        )

        # TODO: update w rate TODO: fix weights in each head
        self.global_weight_conv.data *= new_weight

        self.global_weight_conv = self._secure_tensor(self.global_weight_conv)

        self.global_weight_conv = F.normalize(
            self.global_weight_conv, p=1, dim=(1, 2, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_device = x.device
        return super().forward(x)

    def _check_forward(self, input):
        assert (self.local_weight >= 0).all(), self.local_weight.min()
        assert (input >= 0).all(), input.min()


class AplhaMixerAttentionHeads(nn.Module):
    def __init__(
        self,
        seq_len: int,
        features: int,
        embed_dim: int,
        heads: int,
        n_iterations: int,
        alpha_dynamics_iterations: int,
        output: str,
        hidden_features: int | None = None,
        hidden_seq_len: int | None = None,
        gated: bool = False,
        use_out_proj: bool = True,
        backward_method: str = "fixed_point",
        h_update_rate: float = 1,
        h_softmax_power: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        solver=None,
        convergence_threshold=0,
        normalize_input=True,
        divide_input=False,
        normalize_input_dim=-1,
        normalize_reconstruction=True,
        normalize_reconstruction_dim=-1,
        normalize_h=True,
        normalize_h_dim=-1,
    ):
        super().__init__()
        self.threshold: float = 0.00001
        assert alpha_dynamics_iterations > 0, "Alpha dynamics iterations should be greater than 0"
        self.alpha_dynamics_iterations = alpha_dynamics_iterations
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
        self.divide_input = divide_input
        self.normalize_h = normalize_h
        self.normalize_h_dim = normalize_h_dim
        self.power_softmax = PowerSoftmax(h_softmax_power, dim=self.normalize_h_dim)

        self.embed = nn.Linear(features, embed_dim)

        self.nnmf_layer = NNMFDense(
            in_features= embed_dim // heads, #E
            out_features=self.hidden_features // heads, # D
            n_iterations=n_iterations,
            backward_method=backward_method,
            solver=solver,
            convergence_threshold=convergence_threshold,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=True,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
        )

        self.alpha_init: NonNegativeParameter = NonNegativeParameter(
                torch.ones(1, self.heads, self.hidden_seq_len, self.seq_len), requires_grad=False
        )

        self.gated = gated
        if self.gated:
            self.gate = nn.Linear(features, embed_dim)
            self.gate_activation = nn.SiLU()

        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(embed_dim, features)

        self.save_attn_map = False

    def _prepare_input(self, input):
        if self.normalize_input:
            input = F.normalize(input, p=1, dim=self.normalize_input_dim, eps=1e-20)
        if self.divide_input:
            input = input / input.shape[1]
        return input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            z = self.gate_activation(self.gate(x))
        x = self.embed(x)
        x = torch.clamp(x, min=MINIMUM_POSITIVE)
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        h, reconstruct = self.nnmf_layer(x)
        if self.normalize_h:
            h = F.normalize(h, p=1, dim=self.normalize_h_dim)
        x = self.alpha_dynamics(h, self.nnmf_layer.prepared_input, reconstruct)
        x = x.flatten(-2)
        if self.gated:
            x = x * z
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def alpha_dynamics(self, h, input, h_reconstruction):
        alpha = self.alpha_init.clone().repeat(h.shape[0], 1, 1, 1)
        h_rec_input = h_reconstruction * input
        self.alpha_convergence = []
        for _ in range(self.alpha_dynamics_iterations):
            h_mixed = torch.einsum("bohf,bhoi->bihf", h, alpha)
            alpha_reconstruction = self.nnmf_layer._reconstruct(h_mixed)
            alpha_reconstruction_inv = 1 / (alpha_reconstruction + 1e-20)
            new_alpha = alpha * torch.einsum("bohf,bihf->bhoi", h_rec_input, alpha_reconstruction_inv)
            self.alpha_convergence.append(F.mse_loss(alpha, new_alpha))
            alpha = new_alpha

        return torch.einsum("bohf,bhoi->bihf", h, alpha)

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
