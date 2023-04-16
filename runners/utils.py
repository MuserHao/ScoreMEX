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
        states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{config.sampling.ckpt_id}.pth'),
                            map_location=config.device)
    return states

def load_score(config, states):
    score = get_model(config)
    score = torch.nn.DataParallel(score)

    score.load_state_dict(states[0], strict=True)

    if config.model.ema:
        ema_helper = EMAHelper(mu=self.config.model.ema_rate)
        ema_helper.register(score)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(score)

    score.eval()
    return score

def load_sigmas(config):
    sigmas_th = get_sigmas(config)
    sigmas = sigmas_th.cpu().numpy()
    return sigmas

def get_dataset_and_loader(config, args):
    dataset, _ = get_dataset(args, config)
    dataloader = DataLoader(dataset, batch_size=config.sampling.batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader

# def sample_inpainting(dataloader, score, sigmas):
#     data_iter = iter(dataloader)
#     refer_images, _ = next(data_iter)
#     refer_images = refer_images.to(self.config.device)
#     width = int(np.sqrt(self.config.sampling.batch_size))
#     init_samples = torch.rand(width, width, self.config.data.channels, self.config.data.image_size, self.config.data.image_size, device=self.config.device)
#     init_samples = data_transform(self.config, init_samples)
#     all_samples = anneal_Langevin_dynamics_inpainting(init_samples, refer_images[:width, ...], score, sigmas, self.config.data.image_size, self.config.sampling.n_steps_each, self.config.sampling.step_lr)
#     return refer_images, all_samples

# def save_refer_images(self, refer_images):
#     torch.save(refer_images[:width, ...], os.path.join(self.args.image_folder, 'refer_image.pth'))
#     refer_images = refer_images[:width, None, ...].expand(-1, width, -1, -1, -1).reshape(-1, *refer_images.shape[1:])
#     save_image(refer_images, os.path.join(self.args.image_folder, 'refer_image.png'), nrow=width)
#     return

# def save_inpainting_samples(self, all_samples):
#     for i, sample in enumerate(tqdm.tqdm(all_samples)):
#         sample = sample.view(self.config.sampling.batch_size, self.config.data.channels, self.config.data.image_size, self.config.data.image_size)
#         sample = inverse_data_transform(self.config, sample)
#         image_grid = make_grid(sample, int(np.sqrt(self.config.sampling.batch_size)))
#         save_image(image_grid, os.path.join(self.args.image_folder, 'image_grid_{}.png'.format(i)))
#         torch.save(sample, os.path.join(self.args.image_folder, 'completion_{}.pth'.format(i)))
#     return

# def sample_interpolation(self, init_samples, score, sigmas):
#     all_samples = anneal_Langevin_dynamics_interpolation(init_samples, score, sigmas, self.config.sampling.n_interpolations, self.config.sampling.n_steps_each, self.config.sampling.step_lr, verbose=True, final_only=self.config.sampling.final_only)
#     return all_samples

# def save_interpolation_samples(self, all_samples):
#     if not self.config.sampling.final_only
