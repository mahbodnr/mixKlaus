# mixKlaus
Klaus didn't like the term "anti-Klaus", so, here we go.

# WandB
to log the runs I highly encourage you to use --wandb-api-key flag. You can get your key from [here](https://wandb.ai/authorize). I would also suggest to save it as an environment variable.

# NNMF mixer
A good and fast run:
```
python run.py --model-name nnmf_mixer --num-layers 3 --max-epochs=200 --dataset=c10 --patch=8 --md-iter=5 --head=8 --hidden=96 --off-cls-token --wandb-api-key $WANDB_API_KEY
```