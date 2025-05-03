import wandb
from datetime import datetime
from torch import nn
import numpy as np

from packages.model.loss import LossDetails
from packages.test.test_service import TestResult


def init(arguments, model):
    wandb.init(
        project="partial-motion-capture",
        config=arguments,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    watch_model(model)
    
def watch_model(model):
    wandb.watch(model, nn.MSELoss(), log="all", log_freq=100000)

def unwatch(model):
    wandb.unwatch(model)

def log(epoch, train_loss_list, validation_result: TestResult):
    train_loss = np.mean(train_loss_list)
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': validation_result.loss.get_loss(),
        'val_position_loss': validation_result.loss.position_loss,
        'val_rotation_loss': validation_result.loss.rotation_loss,
        'val_smooth_position_loss': validation_result.loss.smooth_position_loss,
        'val_smooth_rotation_loss': validation_result.loss.smooth_rotation_loss,
        'val_kld_loss': validation_result.loss.kld,
        'val_poss_mse': validation_result.poss_mse,
        'val_l2lq': validation_result.l2lq,
        'val_l2q': validation_result.l2q
    })

def log_train_loss_mid_epoch(loss: float, detail: LossDetails):
    wandb.log({
        'loss_mid_epoch': loss,
        'loss_rotation_mid_epoch': detail.rotation_loss,
        'loss_position_mid_epoch': detail.position_loss,
        'loss_kld_mid_epoch': detail.kld,
        'loss_rotation_smooth_mid_epoch': detail.smooth_rotation_loss,
        'loss_position_smooth_mid_epoch': detail.smooth_position_loss,
    })