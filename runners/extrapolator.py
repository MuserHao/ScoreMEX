import torch
import numpy as np
from sklearn.linear_model import LinearRegression


class Extrapolator:
    def __init__(self, model, data_shape, x_mod, sigmas):
        """
        Initialize an extrapolator object with a regression model, data shape, and sigmas.

        Parameters:
        model (object): The regression model to use for extrapolation.
        data_shape (list or tuple): The shape of the data points.
        x_mod (torch tensor): the x-value score model will be used on.
        sigmas (list): The standard deviations of diffusion (Gaussian) noise level.
        """
        self.model = model
        self.data_shape = data_shape
        self.x_mod = x_mod
        self.sigmas = sigmas
        self.labels = []
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(self.data_shape[0], device=x_mod.device) * c
            labels = labels.long()
            self.labels += [labels]
        self.x = None
        self.y = None
        
    def generate_data(self, scorenet):
        """
        Generate training data using the specified scorenet.

        Parameters:
        data_generator (function): A function or model that generates training data.
                                   Should take two arguments: data_shape and sigmas.
                                   Should return two arrays, x and y.
        """
        #TODO
        self.x, self.y = scorenet(x_mod, labels)
        
    def train_model(self):
        """
        Train the regression model using the generated training data.
        Raises a ValueError if no training data has been generated.
        """
        if self.x is None or self.y is None:
            raise ValueError("Training data not generated")
        self.model.fit(self.x.reshape(-1, 1), self.y)
        
    def make_prediction(self, x_pred):
        """
        Use the trained regression model to make a prediction for the given x values.

        Parameters:
        x_pred (array-like): The x values for which to make predictions.

        Returns:
        array-like: The predicted y values.
        Raises a ValueError if no training data has been generated.
        """
        if self.x is None or self.y is None:
            raise ValueError("Training data not generated")
        return self.model.predict(x_pred.reshape(-1, 1))



      
@torch.no_grad()
def extrapolated_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                noise = torch.randn_like(x_mod)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images
