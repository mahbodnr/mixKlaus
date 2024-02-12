from abc import abstractmethod
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mixKlaus.utils import PowerSoftmax
from nnmf.parameters import NonNegativeParameter

COMPARISSON_TOLERANCE = 1e-5
SECURE_TENSOR_MIN = 1e-5


class NNMFLayer(nn.Module):
    def __init__(
        self,
        n_iterations,
        backward_method="fixed_point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
        solver=None,
        normalize_input=False,
        normalize_input_dim=None,
        normalize_reconstruction=False,
        normalize_reconstruction_dim=None,
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
        if not activate_secure_tensors:
            print(
                "[WARNING] 'activate_secure_tensors' is False! This may lead to numerical instability."
            )

        self.n_iterations = n_iterations
        self.activate_secure_tensors = activate_secure_tensors
        self.return_reconstruction = return_reconstruction
        self.h_update_rate = h_update_rate
        self.keep_h = keep_h

        self.backward = backward_method
        if self.backward == "solver":
            assert solver is not None, "solver must be provided when using solver"
            self.solver = solver
            self.hook = None

        self.normalize_input = normalize_input
        self.normalize_input_dim = normalize_input_dim
        if self.normalize_input and self.normalize_input_dim is None:
            print(
                "[WARNING] normalize_input is True but normalize_input_dim is None! This will normalize the entire input tensor (including batch dimension)"
            )
        self.normalize_reconstruction = normalize_reconstruction
        self.normalize_reconstruction_dim = normalize_reconstruction_dim
        if self.normalize_reconstruction and self.normalize_reconstruction_dim is None:
            print(
                "[WARNING] normalize_reconstruction is True but normalize_reconstruction_dim is None! This will normalize the entire reconstruction tensor (including batch dimension)"
            )
        self.h = None
        self.reconstruction = None
        self.convergence = None

    def _secure_tensor(self, t):
        return t.clamp_min(SECURE_TENSOR_MIN) if self.activate_secure_tensors else t

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
    def _process_reconstruction(self, reconstruction):
        return reconstruction

    @abstractmethod
    def _check_forward(self, input):
        """
        Check that the forward pass is valid
        """

    def _nnmf_iteration(self, input):
        reconstruction = self._reconstruct(self.h)
        reconstruction = self._secure_tensor(reconstruction)
        if self.normalize_reconstruction:
            reconstruction = F.normalize(
                reconstruction, p=1, dim=self.normalize_reconstruction_dim, eps=1e-20
            )
        nnmf_update = input / reconstruction
        assert nnmf_update.isfinite().all() and nnmf_update.isnan().any()==0, (nnmf_update, input, reconstruction, input.isfinite().all(), reconstruction.isfinite().all(), input.isnan().any(), reconstruction.isnan().any(), reconstruction.min(), reconstruction.max())
        new_h = self.h * self._forward(nnmf_update)
        if self.h_update_rate == 1:
            h = new_h
        else:
            h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h), self._process_reconstruction(reconstruction)

    def forward(self, input):
        self.normalize_weights()
        self._check_forward(input)
        if self.normalize_input:
            input = F.normalize(input, p=1, dim=self.normalize_input_dim, eps=1e-20)

        if (not self.keep_h) or (self.h is None):
            self._reset_h(input)

        self.convergence = []
        if self.backward == "all_grads":
            for _ in range(self.n_iterations):
                new_h, self.reconstruction = self._nnmf_iteration(input)
                self.convergence.append(
                    torch.norm(new_h - self.h, p=1, dim=1).mean().item()
                )
                self.h = new_h

        elif self.backward == "fixed_point":
            with torch.no_grad():
                for _ in range(self.n_iterations):
                    new_h, self.reconstruction = self._nnmf_iteration(input)
                    self.convergence.append(
                        torch.norm(new_h - self.h, p=1, dim=1).mean().item()
                    )
                    self.h = new_h

            if self.training:
                new_h, self.reconstruction = self._nnmf_iteration(input)
                self.convergence.append(
                    torch.norm(new_h - self.h, p=1, dim=1).mean().item()
                )
                self.h = new_h

        elif self.backward == "solver":
            with torch.no_grad():
                for _ in range(self.n_iterations):
                    self.h, self.reconstruction = self._nnmf_iteration(input)

            if self.training:
                self.h = self.h.requires_grad_()
                new_h, new_reconstruction = self._nnmf_iteration(input)

                def backward_hook_h(grad):
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

                if self.return_reconstruction:

                    def backward_hook_reconstruction(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        g, self.backward_res = self.solver(
                            lambda y: torch.autograd.grad(
                                new_reconstruction, self.h, y, retain_graph=True
                            )[0]
                            + grad,
                            torch.zeros_like(grad),
                        )
                        return g

                    self.hook_reconstruction = new_reconstruction.register_hook(
                        backward_hook_reconstruction
                    )

                self.hook_h = new_h.register_hook(backward_hook)
                self.h = new_h
                self.reconstruction = new_reconstruction
        else:
            raise NotImplementedError(
                f"backward_method {self.backward} not implemented"
            )

        if self.return_reconstruction:
            return self.h, self.reconstruction
        else:
            return self.h


class NNMFLayerDynamicWeight(NNMFLayer):
    def __init__(
        self,
        n_iterations,
        backward_method="fixed_point",
        h_update_rate=1,
        w_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
        solver=None,
        normalize_input=False,
        normalize_input_dim=None,
        normalize_reconstruction=False,
        normalize_reconstruction_dim=None,
    ):
        super().__init__(
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=return_reconstruction,
            solver=solver,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
        )
        self.w_update_rate = w_update_rate

    def _nnmf_iteration(self, input):
        new_h, new_reconstruction = super()._nnmf_iteration(input)
        self._update_weight(new_h, new_reconstruction, input)
        return new_h, new_reconstruction

    @abstractmethod
    def _update_weight(self, h, input):
        raise NotImplementedError


class NNMFDense(NNMFLayer):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="fixed_point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
            return_reconstruction,
            normalize_input=True,
            normalize_input_dim=1,
            normalize_reconstruction=True,
            normalize_reconstruction_dim=1,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_iterations = n_iterations

        self.weight = NonNegativeParameter(torch.rand(out_features, in_features))
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
        h = self._secure_tensor(h)
        h = F.normalize(F.relu(h), p=1, dim=1)
        return h

    def _check_forward(self, input):
        assert self.weight.sum(1, keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum(1)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=1, dim=1)


class NNMFDenseDynamicWeight(NNMFLayerDynamicWeight, NNMFDense):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="fixed_point",
        h_update_rate=1,
        w_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
    ):
        NNMFDense.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=return_reconstruction,
        )
        self.w_update_rate = w_update_rate

    def _update_weight(self, h, reconstruction, input):
        nnmf_update = input / reconstruction
        new_weight = self.weight.data * F.linear(h.t(), nnmf_update.t())
        self.weight.data = (
            self.w_update_rate * new_weight
            + (1 - self.w_update_rate) * self.weight.data
        )
        self.normalize_weights()


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
        backward_method="fixed_point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
            return_reconstruction,
            normalize_input=True,
            normalize_input_dim=(1, 2, 3),
            normalize_reconstruction=True,
            normalize_reconstruction_dim=(1, 2, 3),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.n_iterations = n_iterations
        self.normalize_channels = normalize_channels
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
        h = self._secure_tensor(h)
        if self.normalize_channels:
            h = F.normalize(F.relu(h), p=1, dim=1)
        else:
            h = self.secure_tensor(h, dim=(1, 2, 3))
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
        backward_method="fixed_point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
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
            return_reconstruction,
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
        if self.h_update_rate == 1:
            h = new_h
        else:
            h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h)


class FPNNMFLayer(NNMFLayer):
    """
    Fixed Point Non-negative Matrix Factorization Layer based on:
    Fixed Point Algorithm for Solving Nonmonotone Variational Inequalities in Nonnegative Matrix Factorization, Hideaki Iiduka, Shizuka Nishino (2016)
    https://arxiv.org/abs/1611.00146
    """

    def _nnmf_iteration(self, input):
        reconstruction = self._reconstruct(self.h)
        if self.normalize_reconstruction:
            reconstruction = F.normalize(
                reconstruction, p=1, dim=self.normalize_reconstruction_dim, eps=1e-20
            )
        reconstruction = self._secure_tensor(reconstruction)
        st_h = self.compute_st_h()
        new_h = F.relu(self.h - st_h * self._forward(reconstruction - input))
        if self.h_update_rate == 1:
            h = new_h
        else:
            h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h), self._process_reconstruction(reconstruction)

    def compute_st_h(self):
        raise NotImplementedError


class FPNNMFLayerDynamicWeight(FPNNMFLayer):
    # TODO: add w_update_rate to init
    def _nnmf_iteration(self, input):
        new_h, new_reconstruction = super()._nnmf_iteration(input)
        self._update_weight(new_h, new_reconstruction, input)
        return new_h, new_reconstruction

    @abstractmethod
    def _update_weight(self, h, input):
        raise NotImplementedError


class FPNNMFDense(FPNNMFLayer, NNMFDense):
    def compute_st_h(self):
        return 2.0 / max(
            1.0, torch.norm(torch.matmul(self.weight.t(), self.weight), "fro")
        )


class FPNNMFDenseDynamicWeight(FPNNMFLayerDynamicWeight, NNMFDenseDynamicWeight):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="fixed_point",
        h_update_rate=1,
        w_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        return_reconstruction=False,
    ):
        NNMFDenseDynamicWeight.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            w_update_rate=w_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=return_reconstruction,
        )

    def compute_st_h(self):
        return 2.0 / max(
            1.0, torch.norm(torch.matmul(self.weight.t(), self.weight), "fro")
        )

    def compute_st_w(self):
        return 2.0 / max(1.0, torch.norm(torch.matmul(self.h.t(), self.h), "fro"))

    def _update_weight(self, h, reconstruction, input):
        st_w = self.compute_st_w()
        new_weight = F.relu(
            self.weight - st_w * torch.matmul(h.t(), reconstruction - input)
        )
        self.weight.data = (
            self.w_update_rate * new_weight
            + (1 - self.w_update_rate) * self.weight.data
        )
        self.normalize_weights()
