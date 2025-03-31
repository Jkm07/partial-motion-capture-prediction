import wandb
from datetime import datetime
from torch import nn
import numpy as np


def init(arguments, model):
    wandb.init(
        project="partial-motion-capture",
        config=arguments,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    watch_model(model)
    
def watch_model(model):
    wandb.watch(model, nn.MSELoss(), log="all", log_freq=100)

def unwatch(model):
    wandb.unwatch(model)

def log(epoch, train_loss_list, mse, mase):
    train_loss = np.mean(train_loss_list)
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_mse': mse,
        'val_mase': mase,
    })