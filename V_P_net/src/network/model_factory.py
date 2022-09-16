from network.lstm import world_p_v_lstm_inte_axis

def get_model(arch, net_config, input_dim=2, output_dim=1):
    if arch == "world_p_v_lstm_axis":
        network = world_p_v_lstm_inte_axis(
            input_dim=input_dim, hidden_dim=[2, 64, 128, 256], num_layer=1,out_dim = output_dim
        )
    else:
        raise ValueError("Invalid architecture: ", arch)
    return network
