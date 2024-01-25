# mixKlaus
Klaus didn't like the term "anti-Klaus", so, here we go.

# WandB
to log the runs I highly encourage you to use --wandb flag. If you are not already logged in, you can also pass your API key with --wandb-api-key flag (NOT RECOMMENDED). You can get your key from [here](https://wandb.ai/authorize).

# NNMF mixer
A good and fast run:
```
python run.py --model-name nnmf_mixer --num-layers 3 --max-epochs=200 --dataset=c10 --patch=8 --md-iter=5 --head=8 --hidden=96 --off-cls-token --wandb-api-key $WANDB_API_KEY
```