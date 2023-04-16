def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)
