from packages.dataloader.dataloader_utils import get_amass_dataloader
from packages.bvhConverter.bvh_converter import get_default_hierarchy
from packages.model.VAE import get_vae_model
import torch
from packages.test import test_service
from packages.utils.subgroups import SUBGROUP_NODES
from packages.math import math_utils
from packages.bvhConverter import amass_service
from packages.utils.slices import POSITION_FLAT, ROTATION_FLAT
from packages.utils.common import print_device_info

@print_device_info
def run(arguments):

    vae = load_model_to_eval(arguments)
    test_data = get_amass_dataloader(arguments.test_dir, arguments.test_batch_size, arguments.sequence_length)
    test_service_instance = test_service.TestService(vae, test_data)

    test_result =  test_service_instance.run_test()
    print(f"General Validation. Result: {test_result}")

    for node_key in SUBGROUP_NODES.keys():
        test_service_instance = test_service.TestService(vae, test_data)
        test_result =  test_service_instance.run_test([node_key])

        print(f"Validation node: {SUBGROUP_NODES[node_key]} . Result: {test_result}")

@print_device_info
def run_save_results(arguments):

    vae = load_model_to_eval(arguments)
    test_data = get_amass_dataloader(arguments.test_dir, 1, arguments.sequence_length)
    test_service_instance = test_service.TestService(vae, test_data)

    with torch.no_grad():
        vae.eval()
        hierarchy = get_default_hierarchy()
        for i, data in enumerate(test_data):
            output, mu, logvar = vae(data)
            result = test_service_instance.run_test_for_given_data(output, data, mu, logvar)
            convert_to_bvh(data[0], hierarchy, get_file_name(i, result.l2lq, True))
            convert_to_bvh(output[0], hierarchy, get_file_name(i, result.l2lq, False))

def convert_to_bvh(data, hierarchy, file_name):
    pos = data[POSITION_FLAT][..., :3].cpu().numpy()
    rot = math_utils.euler_from_quat(data[ROTATION_FLAT]).cpu().numpy()
    amass_service.save_output(pos, rot, hierarchy, file_name)

def get_file_name(idx, l2q, is_original):
    return f"./results/file_l2lq-{l2q}_{idx}" + ("_org" if is_original else "") + ".bvh"

def load_model_to_eval(arguments):
    vae = get_vae_model(arguments)
    vae.load_state_dict(torch.load(arguments.model_path))
    vae.eval()
    return vae