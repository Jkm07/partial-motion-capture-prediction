from packages.dataloader.dataloader_utils import get_amass_dataloader
from packages.bvhConverter.node import get_adjacency_list, add_position_node_to_adjacency_list
from packages.utils.bvh import get_hierarchy
from packages.model.VAE import VAE
import torch
from packages.test import test_service
from packages.model.STGCN import RELATED_NODES
from packages.math import math_utils
from packages.bvhConverter import amass_service

ROTATION_MATRIX_SIZE = 6

def run(arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)

    hierarchy = get_hierarchy()
    adjacency_list = get_adjacency_list(hierarchy[0])
    adjacency_list = add_position_node_to_adjacency_list(adjacency_list)
    vae = VAE(ROTATION_MATRIX_SIZE, arguments.latent_dim, adjacency_list, arguments.sequence_length).to(device)

    vae.load_state_dict(torch.load(arguments.model_path))
    vae.eval()

    test_data = get_amass_dataloader(arguments.test_dir, arguments.test_batch_size, arguments.sequence_length)

    test_service_instance = test_service.TestService(vae, test_data)
    test_result =  test_service_instance.run_test()

    print(f"General Validation. Result: {test_result}")

    for node_key in RELATED_NODES.keys():
        test_service_instance = test_service.TestService(vae, test_data)
        test_result =  test_service_instance.run_test([node_key])

        print(f"Validation node: {RELATED_NODES[node_key]} . Result: {test_result}")

def run_save_results(arguments):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device.type)

    hierarchy = get_hierarchy()
    adjacency_list = get_adjacency_list(hierarchy[0])
    adjacency_list = add_position_node_to_adjacency_list(adjacency_list)
    vae = VAE(ROTATION_MATRIX_SIZE, arguments.latent_dim, adjacency_list, arguments.sequence_length).to(device)

    vae.load_state_dict(torch.load(arguments.model_path))
    vae.eval()

    test_data = get_amass_dataloader(arguments.test_dir, 1, arguments.sequence_length)

    test_service_instance = test_service.TestService(vae, test_data)

    with torch.no_grad():
        vae.eval()
        for i, data in enumerate(test_data):
            output, _, _ = vae(data)
            _, _, l2q = test_service_instance.run_test_for_given_data(output, data)
            convert_to_bvh(data[0], hierarchy, get_file_name(i, l2q, True))
            convert_to_bvh(output[0], hierarchy, get_file_name(i, l2q, False))

def convert_to_bvh(data, hierarchy, file_name):
    pos = data[..., -1, :3].cpu().numpy()
    rot = math_utils.get_euler_from_matrix(data[..., :-1, :]).cpu().numpy()
    amass_service.save_output(pos, rot, hierarchy, file_name)

def get_file_name(idx, l2q, is_original):
    return f"./results/file_l2q-{l2q}_{idx}" + ("_org" if is_original else "") + ".bvh"