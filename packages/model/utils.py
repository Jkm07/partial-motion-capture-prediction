def get_flat_layer_size(in_channels, adjacency_list, seq_len):
        return (seq_len // 12) * in_channels * 4 * len(adjacency_list)