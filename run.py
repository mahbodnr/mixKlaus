import os
import argparse
from pprint import pprint

import torch
import pytorch_lightning as pl
import numpy as np

from mixKlaus.network import Net
from mixKlaus.utils import get_dataloader, get_experiment_name

parser = argparse.ArgumentParser()
parser.add_argument("--comet", action="store_true", dest="use_comet")
parser.add_argument(
    "--comet-api-key", help="API Key for Comet.ml", dest="_comet_api_key", default=None
)
parser.add_argument("--wandb", action="store_true", dest="use_wandb")
parser.add_argument("--wandb-api-key", help="API Key for WandB", dest="_wandb_api_key", default=None)
parser.add_argument(
    "--dataset",
    default="c10",
    type=str,
    choices=["c10", "c100", "svhn", "mnist", "fashionmnist"],
)
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--model-name",
    default="nnmf_mixer",
    type=str,
    choices=[
        "vit",
        "nnmf_mixer",
        "baseline_mixer",
        "capsule_net",
        "cnn",
    ],
)
parser.add_argument("--patch", default=8, type=int)
parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=256, type=int)
parser.add_argument(
    "--optimizer",
    default="adam",
    type=str,
    choices=["adam", "sgd", "madam"],
)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lr-nnmf", default=1e-2, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument(
    "--off-benchmark",
    action="store_false",
    dest="trainer_benchmark",
    help="he value (True or False) to set torch.backends.cudnn.benchmark to. The value for torch.backends.cudnn.benchmark set in the current session will be used (False if not manually set). If deterministic is set to True, this will default to False. You can read more about the interaction of torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic. Setting this flag to True can increase the speed of your system if your input sizes don’t change. However, if they do, then it might make your system slower. The CUDNN auto-tuner will try to find the best algorithm for the hardware when a new input size is encountered. This might also increase the memory usage.",
)
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--precision", default="16-mixed", type=str)
parser.add_argument("--autoaugment", action="store_true")
parser.add_argument("--random_crop_padding", default=4, type=int, help="Padding for random crop augmentation. If 0, no random crop is applied.")
parser.add_argument("--criterion", default="ce", type=str, choices=["ce", "margin"])
parser.add_argument("--label-smoothing", action="store_true")
parser.add_argument("--smoothing", default=0.1, type=float)
parser.add_argument("--rcpaste", action="store_true")
parser.add_argument("--cutmix", action="store_true")
parser.add_argument("--mixup", action="store_true")
parser.add_argument(
    "--md-iter",
    default=7,
    type=int,
    help="Number of iterations in Matrix Decomposition (MD).",
)
parser.add_argument(
    "--router-iter",
    default=3,
    type=int,
    help="Number of iterations in the Capsule Network's Router .",
)
parser.add_argument("--dropout", default=0.0, type=float)
parser.add_argument("--head", default=12, type=int)
parser.add_argument("--num-layers", default=1, type=int)
parser.add_argument("--hidden", default=384, type=int)
parser.add_argument(
    "--mlp-hidden",
    default=384,
    type=int,
    help="Number of hidden units in MLP in encoder blocks.",
)
parser.add_argument(
    "--no-encoder-mlp",
    action="store_false",
    dest="use_encoder_mlp",
    help="Disable MLP in encoder blocks.",
)
parser.add_argument(
    "--use-conv",
    action="store_true",
    help="Use convolutions for NNMF Mixer global weights.",
)
parser.add_argument("--kernel-size", default=3, type=int)
parser.add_argument("--stride", default=1, type=int)
parser.add_argument("--padding", default=0, type=int, dest="conv_padding")
parser.add_argument(
    "--nnmf_backward",
    type=str,
    default="all_grads",
    choices=["all_grads", "fixed_point", "solver"],
    help="How to compute gradients for NNMF Layers.",
)
parser.add_argument("--no-pos-emb", action="store_false", dest="pos_emb", help="desable positional embedding in the Transformer")
parser.add_argument("--off-cls-token", action="store_false", dest="is_cls_token", help="desable the <CLS> token in the Transformer")
parser.add_argument("--output-mode", default="mean", type=str, choices=["mean", "fc"], help="How to compute the output of the Transformer ONLY IF the <CLS> token is off")
parser.add_argument(
    "--matmul-precision",
    default="medium",
    type=str,
    choices=["medium", "high", "highest"],
)
parser.add_argument("--log-gradients", action="store_true", help="Save gradients during training.")
parser.add_argument("--log-gradients-interval", default=250, type=int)
parser.add_argument("--no-log-weights", action="store_false", dest="log_weights", help="Disable logging weights during training.")
parser.add_argument("--model-summary-depth", default=-1, type=int)
parser.add_argument("--tags", default="", type=str, help="Comma separated tags.")
parser.add_argument("--seed", default=2045, type=int)  # Singularity is near
parser.add_argument("--project-name", default="mixKlaus", type=str)
parser.add_argument("--cnn-features", default=[64, 128], nargs="+", type=int, help="Number of features in CNN layers of the CNN model.")
parser.add_argument("--ann-layers", default=[256, 128, 10], nargs="+", type=int, help="Number of features in ANN layers of the CNN model.")
parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
parser.add_argument("--allow-download", action="store_true", dest="download_data")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_float32_matmul_precision(args.matmul_precision)
args.gpus = torch.cuda.device_count()
args.num_workers = 4 * args.gpus if args.gpus else 8
if not args.gpus:
    args.precision = 32


args.seq_len = args.patch**2 + 1 if args.is_cls_token else args.patch**2

train_dl, test_dl = get_dataloader(args)
args._sample_input_data = next(iter(train_dl))[0][0:10].to(
    "cuda" if args.gpus else "cpu"
)

if __name__ == "__main__":
    print("Arguments:")
    pprint({k: v for k, v in vars(args).items() if not k.startswith("_")})
    experiment_name = get_experiment_name(args)
    args.experiment_name = experiment_name
    print(f"Experiment: {experiment_name}")
    if args.use_comet:
        if args.use_wandb:
            print("[WARNING] Both Comet.ml and WandB are enabled. Using Comet.ml.")
        print("[INFO] Log with Comet.ml!")
        logger = pl.loggers.CometLogger(
            api_key=args._comet_api_key,
            save_dir="logs",
            project_name=args.project_name,
            experiment_name=experiment_name,
        )
        del args._comet_api_key # remove the API key from args
    elif args.use_wandb:
        print("[INFO] Log with WandB!")
        import wandb

        wandb.login(key=args._wandb_api_key)
        logger = pl.loggers.WandbLogger(
            log_model="all",
            save_dir="logs",
            project=args.project_name,
            name=experiment_name,
        )
        del args._wandb_api_key # remove the API key from args
    else:
        print("[INFO] Log with CSV")
        logger = pl.loggers.CSVLogger(save_dir="logs", name=experiment_name)

    net = Net(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename=experiment_name + "-{epoch:03d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    if args.profile:
        print("[INFO] Profiling!")
        profiler = pl.profilers.PyTorchProfiler(
            output_filename=f"logs/profile/{experiment_name}.txt",
            # use_cuda=True,
            profile_memory=True,
            export_to_chrome=True,
            use_cpu=False,
        )
    else:
        profiler = None
        
    trainer = pl.Trainer(
        precision=args.precision,
        fast_dev_run=args.dry_run,
        accelerator="auto",
        devices=args.gpus if args.gpus else "auto",
        benchmark=args.trainer_benchmark,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        enable_model_summary=False,  # Implemented seperately inside the Trainer
        profiler=profiler,
    )
    trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=test_dl)

    # Save model
    save_models_dir = "models"
    save_model_path = os.path.join(save_models_dir, experiment_name + ".ckpt")
    trainer.save_checkpoint(save_model_path)
    print(f"Model saved to {save_model_path}")
    # add model to comet
    if args._comet_api_key:
        urls = logger.experiment.log_model(
            experiment_name, save_model_path, overwrite=True
        )
        print(f"Model saved to comet: {urls}")