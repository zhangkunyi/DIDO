from network.model_resnet import q_ResNet1D_test9, BasicBlock1D

# todo change training model
def get_model(arch, net_config, input_dim=3, output_dim=3):
    if arch in ["resnet"]:
        network = q_ResNet1D_test9(
            BasicBlock1D, input_dim, output_dim, [2, 2, 2, 2], net_config["in_dim"]
        )
    else:
        raise ValueError("Invalid architecture in model_factory.py: ", arch)
    return network
