import argparse


def get_train_arguments():
    parser = argparse.ArgumentParser(description="Train main model")

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--no_improvment_stop", type=int, default=3)

    parser.add_argument("--train_dir", type=str, default="datasets/amass/run_test_one_sub")
    parser.add_argument("--valid_dir", type=str, default="datasets/amass/run_test")
    parser.add_argument("--test_dir", type=str, default="datasets/amass/run_test")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")

    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=32)

    parser.add_argument("--latent_dim", type=int, default=2024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sequence_length", type=int, default=60)

    return parser.parse_args()