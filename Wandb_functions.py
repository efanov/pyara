from Wandb_token import WANDB_TOKEN

from config import CFG
import wandb

def wandb_login():
    try:
        wandb.login(key=WANDB_TOKEN)
        anonymous = None
    except:
        anonymous = "must"
        print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


def wandb_init():
    run = wandb.init(project=CFG.wandb_project_name,
                          anonymous = None,
                          name=CFG.wandb_run_name,
                         )

    wandb.config.epochs = CFG.epochs
    wandb.config.train_batch_size = CFG.train_bs
    wandb.config.valid_batch_sizw = CFG.valid_bs
    wandb.config.samples = CFG.num_item_all
    wandb.config.Debug = CFG.DEBUG
    wandb.config.sr = CFG.SAMPLE_RATE
    wandb.config.lr = CFG.lr
    return run