from packages.dataloader.dataloader_utils import get_amass_dataloader
from packages.model.varational_autoencoder import VAE, vae_loss
from packages.utils.bvh import get_hierarchy
from packages.utils import joint_utils
from packages.utils import wandb_utils
from packages.bvhConverter.node import get_adjacency_list, add_position_node_to_adjacency_list
from packages.test import test_service
import torch
from torchinfo import summary
import os
import datetime

ROTATION_MATRIX_SIZE = 6

def run(arguments):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)

    hierarchy = get_hierarchy()
    adjacency_list = get_adjacency_list(hierarchy[0])
    adjacency_list = add_position_node_to_adjacency_list(adjacency_list)
    
    vae = VAE(ROTATION_MATRIX_SIZE, arguments.latent_dim, adjacency_list, arguments.sequence_length).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=arguments.learning_rate)

    #summary(vae, input_size=(8, 60, 52, 6))

    train_data = get_amass_dataloader(arguments.train_dir, arguments.train_batch_size)
    valid_data = get_amass_dataloader(arguments.valid_dir, arguments.valid_batch_size)
    #test_data = get_amass_dataloader(arguments.test_dir, arguments.test_batch_size)

    wandb_utils.init(arguments, vae)

    test_service_instance = test_service.TestService(vae, valid_data)

    for epoch in range(arguments.epoch):
        batch_losses = []
        for i, batch in enumerate(train_data):
            dropout_batch, disable_joint_indexes = joint_utils.input_dropout(batch)

            optimizer.zero_grad()
            vae.train()
            recon_batch, mu, logvar = vae(dropout_batch)

            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss))
            if i % 100 == 0:
                print(f'Loss {float(loss)}')

        valid_test, shouldStop = validation(vae, test_service_instance, epoch, arguments)
        wandb_utils.log(epoch, batch_losses, valid_test['mse'], valid_test['mase'])
        
        if shouldStop:
            break

def validation(model: torch.nn.Module, test_service_instance: test_service.TestService, epoch, arguments) -> bool:
    valid_test = test_service_instance.run_test()
    print(f"Validation {valid_test}")

    if(test_service_instance.is_last_test_improve_result()):
        print(f"Save {epoch} epoch model as best result - {valid_test}")
        save_model(model, epoch, valid_test['mse'], valid_test['mase'], arguments)

    if(epoch - test_service_instance.get_idx_of_last_best_result() > arguments.no_improvment_stop):
        print(f"Stop on {epoch} becouse lack of improvment through last {arguments.no_improvment_stop} epochs")
        return valid_test, True
    
    return valid_test, False

def save_model(model: torch.nn.Module, epoch, loss, mase, arguments):
    date = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
    path = os.path.join(arguments.checkpoint_dir, f'model_epoch_{epoch}_loss_{loss}_date_{date}_mase_{float(mase)}')
    wandb_utils.unwatch(model)
    torch.save(model, path)
    wandb_utils.watch_model(model)
