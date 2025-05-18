import argparse


def get_train_arguments():
    parser = argparse.ArgumentParser(description="Train main model")

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--no_improvment_stop", type=int, default=3)
    parser.add_argument("--save_epoch_skip", type=int, default=1)

    parser.add_argument("--train_dir", type=str, default="datasets/amass/run_test_one_sub")
    parser.add_argument("--valid_dir", type=str, default="datasets/amass/run_test_one_sub")
    parser.add_argument("--test_dir", type=str, default="datasets/amass/run_test_one_sub")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=4096)

    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sequence_length", type=int, default=64)

    parser.add_argument("--model_path", type=str, default="checkpoints/storage/model_work")

    parser.add_argument("--data_representation", type=str, default="rotation_matrix", choices=["quaternion", "rotation_matrix"],)

    return parser.parse_args()