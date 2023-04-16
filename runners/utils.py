import torch
from datasets import get_dataset
from models.ema import EMAHelper
from models import get_sigmas
from torch.utils.data import DataLoader


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)


def load_states(config, args):
    if config.sampling.ckpt_id is None:
        states = torch.load(os.path.join(args.log_path, 'checkpoint.pth'), map_location=config.device)
    else:
        states = torch.load(os.path.join(args.log_path, f'checkpoint_{config.sampling.ckpt_id}.pth'),
                            map_location=config.device)
    return states


def load_score(config, states):
    score = get_model(config)
    score = torch.nn.DataParallel(score)

    score.load_state_dict(states[0], strict=True)

    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(score)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(score)
    return score


def get_dataset_and_loader(config, args):
    dataset, _ = get_dataset(args, config)
    dataloader = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader
