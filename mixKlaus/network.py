import time
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt

from mixKlaus.augmentation import CutMix, MixUp
from mixKlaus.utils import (
    get_model,
    get_criterion,
    get_layer_outputs,
    get_experiment_tags,
)
from mixKlaus.lr_scheduler import GradualWarmupScheduler, StopScheduler
from mixKlaus.utils import get_sparsity
from nnmf.parameters import NonNegativeParameter
from nnmf.modules import NNMFLayer


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams.update(vars(hparams))
        self.save_hyperparameters(
            ignore=[key for key in self.hparams.keys() if key[0] == "_"]
        )
        self.model = get_model(hparams)
        self.criterion = get_criterion(hparams)
        if hparams.cutmix:
            self.cutmix = CutMix(hparams.size, beta=1.0)
        if hparams.mixup:
            self.mixup = MixUp(alpha=1.0)
        self.log_image_flag = hparams._comet_api_key is None
    
    def log_time(func):
        """
        A decorator to measure the time of a function and log it.
        """
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            self.log(f"{func.__name__}_time", end - start)
            return result

        return wrapper
        
    @log_time
    def forward(self, x):
        return self.model(x)

    def calculate_loss(self, out, label):
        return self.criterion(out, label)

    def configure_optimizers(self):
        nnmf_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if isinstance(param, NonNegativeParameter):
                nnmf_params.append(param)
            else:
                other_params.append(param)
        print(
            f"Optimizer params:\nNNMF parameters: {len(nnmf_params)}\nOther parameters: {len(other_params)}"
        )

        if self.hparams.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                params=[
                    {
                        "params": other_params,
                        "lr": self.hparams.lr,
                        "initial_lr": self.hparams.lr,
                    },
                    {
                        "params": nnmf_params,
                        "lr": self.hparams.lr_nnmf,
                        "initial_lr": self.hparams.lr_nnmf,
                    },
                ],
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=[
                    {
                        "params": other_params,
                        "lr": self.hparams.lr,
                        "initial_lr": self.hparams.lr,
                    },
                    {
                        "params": nnmf_params,
                        "lr": self.hparams.lr_nnmf,
                        "initial_lr": self.hparams.lr_nnmf,
                    },
                ],
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                params=[
                    {
                        "params": other_params,
                        "lr": self.hparams.lr,
                        "initial_lr": self.hparams.lr,
                    },
                    {
                        "params": nnmf_params,
                        "lr": self.hparams.lr_nnmf,
                        "initial_lr": self.hparams.lr_nnmf,
                    },
                ],
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "madam":
            from nnmf.optimizer import Madam

            self.optimizer = Madam(
                params=[
                    {
                        "params": other_params,
                        "lr": self.hparams.lr,
                        "initial_lr": self.hparams.lr,
                    },
                    {
                        "params": nnmf_params,
                        "lr": self.hparams.lr_nnmf,
                        "initial_lr": self.hparams.lr_nnmf,
                        "nnmf": True,
                        "foreach": False,
                    },
                ],
                betas=(self.hparams.beta1, self.hparams.beta2),
                weight_decay=self.hparams.weight_decay,
            )

        else:
            raise NotImplementedError(f"Unknown optimizer: {self.hparams.optimizer}")

        if self.hparams.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.max_epochs
                if self.hparams.lr_scheduler_T_max is None
                else self.hparams.lr_scheduler_T_max,
                eta_min=self.hparams.min_lr,
                last_epoch=self.hparams.lr_scheduler_last_epoch,
            )
        elif self.hparams.lr_scheduler == "cosine_restart":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.lr_scheduler_T_0,
                T_mult=self.hparams.lr_scheduler_T_mult,
                eta_min=self.hparams.min_lr,
            )
        elif self.hparams.lr_scheduler.lower() == "none":
            return self.optimizer
        else:
            raise NotImplementedError(
                f"Unknown lr_scheduler: {self.hparams.lr_scheduler}"
            )
        if self.hparams.lr_warmup_epochs > 0:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1.0,
                total_epoch=self.hparams.lr_warmup_epochs,
                after_scheduler=self.scheduler,
            )
        if self.hparams.lr_scheduler_stop_epoch is not None:
            self.scheduler = StopScheduler(
                self.optimizer,
                base_scheduler=self.scheduler,
                stop_epoch=self.hparams.lr_scheduler_stop_epoch,
            )
        return [self.optimizer], [self.scheduler]

    def on_fit_start(self):
        summary = pl.utilities.model_summary.ModelSummary(
            self, max_depth=self.hparams.model_summary_depth
        )
        if hasattr(self.logger.experiment, "log_asset_data"):
            self.logger.experiment.log_asset_data(
                str(summary), file_name="model_summary.txt"
            )
        print(summary)

    def on_train_start(self):
        # wandb watch model
        if isinstance(self.logger, pl.loggers.WandbLogger):
            log = {
                    (True, False): "gradients",
                    (True, True): "all",
                    (False, True): "parameters",
                    (False, False): None,
                }[(self.hparams.log_gradients, self.hparams.log_weights)]
            print(f"[INFO] WandB watch log: {log}")
            self.logger.watch(
                self.model,
                log=log,
            )
        # Number of parameters:
        self.log(
            "trainable_params",
            float(sum(p.numel() for p in self.model.parameters() if p.requires_grad)),
        )
        self.log("total_params", float(sum(p.numel() for p in self.model.parameters())))

        # Tags: #TODO: add for wandb
        tags = self.hparams.tags.split(",")
        tags = [tag.strip() for tag in tags]
        if hasattr(self.logger.experiment, "add_tags"):
            # arg parser tags
            if not (tags[0] == "" and len(tags) == 1):
                self.logger.experiment.add_tags(tags)
            # default tags:
            self.logger.experiment.add_tags(get_experiment_tags(self.hparams))

    def _step(self, img, label):
        if self.hparams.cutmix or self.hparams.mixup:
            if self.hparams.cutmix:
                img, label, rand_label, lambda_ = self.cutmix((img, label))
            elif self.hparams.mixup:
                if torch.rand(1).item() <= 0.8:
                    img, label, rand_label, lambda_ = self.mixup((img, label))
                else:
                    img, label, rand_label, lambda_ = (
                        img,
                        label,
                        torch.zeros_like(label),
                        1.0,
                    )

            out = self(img)
            loss = (self.calculate_loss(out, label) * lambda_) + (
                self.calculate_loss(out, rand_label) * (1.0 - lambda_)
            )
        else:
            out = self(img)
            loss = self.calculate_loss(out, label)

        return out, loss

    def training_step(self, batch, batch_idx):
        img, label = batch
        out, loss = self._step(img, label)

        if not self.log_image_flag and not self.hparams.dry_run:
            self.log_image_flag = True
            self._log_image(img.clone().detach().cpu())

        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("loss", loss)
        self.log("acc", acc)

        return loss

    def on_train_epoch_end(self):
        # log NNMF convergence
        if self.hparams.log_nnmf_convergence:
            if self.hparams.use_wandb:
                for name, module in self.model.named_modules():
                    if isinstance(module, NNMFLayer):
                        # h convergence
                        fig = plt.figure()
                        plt.plot(torch.tensor(module.convergence).detach().cpu().numpy())
                        wandb.log({f"h_convergence/{name}": fig})
                        # Reconstruction mse
                        fig = plt.figure()
                        plt.plot(torch.tensor(module.convergence).detach().cpu().numpy())
                        wandb.log({f"reconstruction_mse/{name}": fig})


        # log learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True)
        # check if there is any nan value in model parameters
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"[ERROR] {name} has nan value. Training stopped.")
        # log weights and layer outputs
        if self.hparams.log_weights:
            if not self.hparams.dry_run and hasattr(
                self.logger.experiment, "log_histogram_3d"
            ):
                # log the output of each layer #TODO: add for wandb
                layer_outputs = get_layer_outputs(
                    self.model, self.hparams._sample_input_data
                )
                for name, output in layer_outputs.items():
                    try:
                        self.logger.experiment.log_histogram_3d(
                            output.detach().cpu(),
                            name=name + ".output",
                            epoch=self.current_epoch,
                        )
                    except IndexError:
                        # Values closer than 1e-20 to zerro will lead to index error
                        positive_output = output[output > 0]
                        pos_min = (
                            positive_output.min().item()
                            if positive_output.numel() > 0
                            else float("inf")
                        )
                        negative_output = output[output < 0]
                        neg_min = (
                            abs(negative_output.max().item())
                            if negative_output.numel() > 0
                            else float("inf")
                        )
                        self.logger.experiment.log_histogram_3d(
                            output.detach().cpu(),
                            name=name + ".output",
                            epoch=self.current_epoch,
                            start=min(pos_min, neg_min),
                        )
                    except Exception as e:
                        raise e
                # log weights #TODO: add for wandb
                for name, param in self.model.named_parameters():
                    try:
                        self.logger.experiment.log_histogram_3d(
                            param.detach().cpu(),
                            name=name,
                            epoch=self.current_epoch,
                        )
                    except IndexError:
                        # Values closer than 1e-20 to zerro will lead to index error
                        positive_param = param[param > 0]
                        pos_min = (
                            positive_param.min().item()
                            if positive_param.numel() > 0
                            else float("inf")
                        )
                        negative_param = param[param < 0]
                        neg_min = (
                            abs(negative_param.max().item())
                            if negative_param.numel() > 0
                            else float("inf")
                        )
                        self.logger.experiment.log_histogram_3d(
                            param.detach().cpu(),
                            name=name,
                            epoch=self.current_epoch,
                            start=min(pos_min, neg_min),
                        )
        # log weight sparsity
        if self.hparams.log_sparsity:
            for name, param in self.model.named_parameters():
                if isinstance(param, NonNegativeParameter):
                    self.log(
                        name=f"sparsity/{name}",
                        value= get_sparsity(param, dim= None)
                    )
                    for dim in range(param.dim()):
                        self.log(
                        name=f"sparsity/{name}_dim{dim}",
                        value= get_sparsity(param, dim= dim) 
                        )

    # def backward(self, loss):
    #     loss.backward(retain_graph=True)

    def on_before_optimizer_step(self, optimizer):
        # log gradients once per epoch #TODO: add for wandb
        if (
            self.hparams.log_gradients
            and hasattr(self.logger.experiment, "log_histogram_3d")
            and not self.hparams.dry_run
            and self.trainer.global_step % self.hparams.log_gradients_interval == 0
        ):
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                try:
                    self.logger.experiment.log_histogram_3d(
                        param.grad.detach().cpu(),
                        name=name + ".grad",
                        epoch=self.current_epoch,
                        step=self.trainer.global_step,
                    )
                except IndexError:
                    # Values closer than 1e-20 to zerro will lead to index error
                    positive_grad = param.grad[param.grad > 0]
                    pos_min = (
                        positive_grad.min().item()
                        if positive_grad.numel() > 0
                        else float("inf")
                    )
                    negative_grad = param.grad[param.grad < 0]
                    neg_min = (
                        abs(negative_grad.max().item())
                        if negative_grad.numel() > 0
                        else float("inf")
                    )
                    self.logger.experiment.log_histogram_3d(
                        param.grad.detach().cpu(),
                        name=name + ".grad",
                        epoch=self.current_epoch,
                        step=self.trainer.global_step,
                        start=min(pos_min, neg_min),
                    )
                except Exception as e:
                    raise e

    def on_train_batch_end(self, out, batch, batch_idx):
        for name, module in self.model.named_modules():
            # Renormalize NNMF parameters
            if hasattr(module, "normalize_weights"):
                module.normalize_weights()
            if hasattr(module, "forward_iterations"):
                if module.convergence_threshold > 0:
                    self.log(
                        f"forward_iterations/{name}",
                        module.forward_iterations,
                    )

    def validation_step(self, batch, batch_idx):
        img, label = batch
        out = self(img)
        loss = self.calculate_loss(out, label)
        acc = torch.eq(out.argmax(-1), label).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def _log_image(self, image):
        # TODO
        pass

    def log_free_mem(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
        self.log("free_memory", free_memory)

    def on_train_end(self):
        if hasattr(self.logger.experiment, "end"):
            self.logger.experiment.end()
