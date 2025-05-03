from packages.dataloader.dataloader_utils import get_amass_dataloader
from packages.model.VAE import get_vae_model
from packages.model.loss import vae_loss
from packages.utils import joint_utils
from packages.utils import wandb_utils
from packages.utils.common import print_device_info
from packages.test import test_service
from packages.test.test_service import TestResult
import torch
import os
import datetime

@print_device_info
def run(arguments):
    vae = get_vae_model(arguments)
    optimizer = torch.optim.Adam(vae.parameters(), lr=arguments.learning_rate)

    train_data = get_amass_dataloader(arguments.train_dir, arguments.train_batch_size, arguments.sequence_length)
    valid_data = get_amass_dataloader(arguments.valid_dir, arguments.valid_batch_size, arguments.sequence_length)

    wandb_utils.init(arguments, vae)

    test_service_instance = test_service.TestService(vae, valid_data)

    for epoch in range(arguments.epoch):
        batch_losses = []
        for i, batch in enumerate(train_data):
            vae.train()
            dropout_batch, disable_joint_indexes = joint_utils.input_dropout(batch)
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(dropout_batch)

            loss, detail = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss))
            if i % 100 == 0:
                wandb_utils.log_train_loss_mid_epoch(float(loss), detail)
                print(f'Loss {float(loss)}. ROT: {detail.rotation_loss} POS: {detail.position_loss} KLD: {detail.kld}, SMOOTH_ROT {detail.smooth_rotation_loss}, SMOOTH_POS {detail.smooth_position_loss}')

        valid_test, shouldStop = validation(vae, test_service_instance, epoch, arguments)
        wandb_utils.log(epoch, batch_losses, valid_test)
        
        if shouldStop:
            break

def validation(model: torch.nn.Module, test_service_instance: test_service.TestService, epoch, arguments) -> tuple[TestResult, bool]:
    valid_test = test_service_instance.run_test()
    print(f"Validation {valid_test}")

    if(epoch % arguments.save_epoch_skip != 0):
        return valid_test, False

    if(test_service_instance.is_last_test_improve_result(skip_epoch=arguments.save_epoch_skip)):
        print(f"Save {epoch} epoch model as best result - {valid_test}")
        save_model(model, epoch, valid_test.loss.get_loss(), valid_test.l2lq, arguments)

    if(epoch // arguments.save_epoch_skip - test_service_instance.get_idx_of_last_best_result(skip_epoch=arguments.save_epoch_skip) > arguments.no_improvment_stop):
        print(f"Stop on {epoch} becouse lack of improvment through last {arguments.no_improvment_stop} epochs")
        return valid_test, True
    
    return valid_test, False

def save_model(model: torch.nn.Module, epoch, loss, l2q, arguments):
    date = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
    path = os.path.join(arguments.checkpoint_dir, f'model_epoch_{epoch}_loss_{loss}_date_{date}_l2lq_{float(l2q)}.pth')
    wandb_utils.unwatch(model)
    torch.save(model.state_dict(), path)
    wandb_utils.watch_model(model)
