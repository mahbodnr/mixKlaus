from abc import abstractmethod
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mixKlaus.utils import PowerSoftmax
from nnmf.parameters import NonNegativeParameter

COMPARISSON_TOLERANCE = 1e-5


class NNMFLayer(nn.Module):
    def __init__(
        self,
        n_iterations,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        solver=None,
    ):
        super().__init__()
        assert n_iterations >= 0 and isinstance(
            n_iterations, int
        ), f"n_iterations must be a positive integer, got {n_iterations}"
        assert (
            0 < h_update_rate <= 1
        ), f"h_update_rate must be in (0,1], got {h_update_rate}"
        assert backward_method in [
            "fixed_point",
            "solver",
            "all_grads",
        ], f"backward_method must be one of 'fixed_point', 'solver', 'all_grads', got {backward_method}"

        self.n_iterations = n_iterations
        self.activate_secure_tensors = activate_secure_tensors
        self.h_update_rate = h_update_rate
        self.keep_h = keep_h
        self.sparsity = PowerSoftmax(sparsity_rate, dim=1)

        self.backward = backward_method
        if self.backward == "solver":
            assert solver is not None, "solver must be provided when using solver"
            self.solver = solver
            self.hook = None

        self.h = None
        self.normalize_dim = None

    def secure_tensor(self, t):
        if not self.activate_secure_tensors:
            return t
        assert self.normalize_dim is not None, "normalize_dim must be set"
        return F.normalize(F.relu(t), p=1, dim=self.normalize_dim, eps=1e-20)

    @abstractmethod
    def normalize_weights(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_h(self, x):
        raise NotImplementedError

    @abstractmethod
    def _reconstruct(self, h):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, nnmf_update):
        raise NotImplementedError

    @abstractmethod
    def _process_h(self, h):
        raise NotImplementedError

    @abstractmethod
    def _check_forward(self, input):
        """
        Check that the forward pass is valid
        """

    def _nnmf_iteration(self, input):
        X_r = self._reconstruct(self.h)
        # X_r = self.secure_tensor(X_r)
        X_r = F.normalize(X_r.clamp_min(0.0001), p=1, dim=self.normalize_dim, eps=1e-20)
        nnmf_update = input / (X_r + 1e-12)
        new_h = self.h * self._forward(nnmf_update)
        h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h)

    def forward(self, input):
        assert self.normalize_dim is not None, "normalize_dim must be set"

        self.normalize_weights()
        self._check_forward(input)
        input = F.normalize(input, p=1, dim=self.normalize_dim, eps=1e-20)

        if (not self.keep_h) or (self.h is None):
            self._reset_h(input)

        if self.backward == "all_grads":
            for _ in range(self.n_iterations):
                self.h = self._nnmf_iteration(input)

        elif self.backward == "fixed_point" or self.backward == "solver":
            with torch.no_grad():
                for _ in range(self.n_iterations - 1):
                    self.h = self._nnmf_iteration(input)

            if self.training:
                if self.backward == "solver":
                    self.h = self.h.requires_grad_()
                new_h = self._nnmf_iteration(input)
                if self.backward == "solver":

                    def backward_hook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        g, self.backward_res = self.solver(
                            lambda y: torch.autograd.grad(
                                new_h, self.h, y, retain_graph=True
                            )[0]
                            + grad,
                            torch.zeros_like(grad),
                        )
                        return g

                    self.hook = new_h.register_hook(backward_hook)
                self.h = new_h
        return self.h


class NNMFDense(NNMFLayer):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_iterations = n_iterations

        self.weight = NonNegativeParameter(torch.rand(out_features, in_features))
        self.normalize_dim = 1
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=1)

    def _reset_h(self, x):
        self.h = F.normalize(torch.ones(x.shape[0], self.out_features), p=1, dim=1).to(
            x.device
        )

    def _reconstruct(self, h):
        return F.linear(h, self.weight.t())

    def _forward(self, nnmf_update):
        return F.linear(nnmf_update, self.weight)

    def _process_h(self, h):
        h = self.secure_tensor(h)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def _check_forward(self, input):
        assert self.weight.sum(0, keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum(0)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=1, dim=0)


class NNMFConv2d(NNMFLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_iterations,
        padding=0,
        stride=1,
        dilation=1,
        normalize_channels=False,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.n_iterations = n_iterations
        self.normalize_channels = normalize_channels
        self.normalize_dim = (1, 2, 3)
        if self.dilation != (1, 1):
            raise NotImplementedError(
                "Dilation not implemented for NNMFConv2d, got dilation={self.dilation}"
            )

        self.weight = NonNegativeParameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))

    def _reconstruct(self, h):
        return F.conv_transpose2d(
            h,
            self.weight,
            padding=self.padding,
            stride=self.stride,
        )

    def _forward(self, nnmf_update):
        return F.conv2d(
            nnmf_update, self.weight, padding=self.padding, stride=self.stride
        )

    def _process_h(self, h):
        if self.normalize_channels:
            h = F.normalize(F.relu(h), p=1, dim=1)
        else:
            h = self.secure_tensor(h)
        return h

    def _reset_h(self, x):
        output_size = [
            (x.shape[-2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]
            + 1,
            (x.shape[-1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]
            + 1,
        ]
        self.h = torch.ones(x.shape[0], self.out_channels, *output_size).to(x.device)

    def _check_forward(self, input):
        assert self.weight.sum((1, 2, 3), keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum((1, 2, 3))
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()


class NNMFVJP(NNMFLayer):
    def _nnmf_iteration(self, input):
        if isinstance(self.h, tuple):
            reconstruct = self._reconstruct(*self.h)
        else:
            reconstruct = self._reconstruct(self.h)
        reconstruct = self.secure_tensor(reconstruct)
        nnmf_update = input / (reconstruct + 1e-20)
        h_update = torch.autograd.functional.vjp(
            self._reconstruct,
            self.h,
            nnmf_update,
            create_graph=True,
        )[1]
        if isinstance(self.h, tuple):
            new_h = tuple(
                self.h_update_rate * h_update[i] * self.h[i]
                + (1 - self.h_update_rate) * self.h[i]
                for i in range(len(self.h))
            )
        else:
            new_h = (
                self.h_update_rate * h_update * self.h
                + (1 - self.h_update_rate) * self.h
            )
        return self._process_h(new_h)


class NNMFDenseVJP(NNMFVJP, NNMFDense):
    """
    NNMFDense with VJP backward method
    """


class NNMFConv2dVJP(NNMFVJP, NNMFConv2d):
    """
    NNMFConv2d with VJP backward method
    """


class NNMFConvTransposed2d(NNMFConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_iterations,
        padding=0,
        output_padding=0,
        stride=1,
        dilation=1,
        normalize_channels=False,
        backward_method="fixed point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            n_iterations,
            padding,
            stride,
            dilation,
            normalize_channels,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.output_padding = _pair(output_padding)
        assert (
            (self.output_padding[0] < self.stride[0])
            or (self.output_padding[0] < self.dilation[0])
        ) and (
            (self.output_padding[1] < self.stride[1])
            or (self.output_padding[1] < self.dilation[1])
        ), f"RuntimeError: output padding must be smaller than either stride or dilation, but got output_padding={self.output_padding}, stride={self.stride}, dilation={self.dilation}"

        self.weight = NonNegativeParameter(
            torch.rand(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.reset_parameters()

    def _reset_h(self, x):
        output_size = [
            (x.shape[-2] - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.kernel_size[0]
            + self.output_padding[0],
            (x.shape[-1] - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.kernel_size[1]
            + self.output_padding[1],
        ]
        self.h = torch.ones(x.shape[0], self.out_channels, *output_size).to(x.device)

    def _nnmf_iteration(self, input, h):
        X_r = F.conv2d(h, self.weight, padding=self.padding, stride=self.stride)
        X_r = self.secure_tensor(X_r, dim=(1, 2, 3))
        if X_r.shape != input.shape:
            input = F.pad(
                input,
                [
                    0,
                    X_r.shape[-1] - input.shape[-1],
                    0,
                    X_r.shape[-2] - input.shape[-2],
                ],
            )
        nnmf_update = input / (X_r + 1e-20)
        new_h = h * F.conv_transpose2d(
            nnmf_update,
            self.weight,
            padding=self.padding,
            stride=self.stride,
            output_padding=self.output_padding,
        )
        h = self.h_update_rate * new_h + (1 - self.h_update_rate) * h
        if self.normalize_channels:
            # h = F.normalize(F.relu(h), p=1, dim=1)
            h = self.sparsity(F.relu(h))
        else:
            h = self.secure_tensor(h, dim=(1, 2, 3))
        return h
