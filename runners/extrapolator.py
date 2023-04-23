import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import CubicSpline
from abc import ABC, abstractmethod

class Extrapolator(ABC):
    def __init__(self, data_shape, x_mod, sigmas):
        self.data_shape = data_shape
        self.x_mod = x_mod
        self.sigmas = sigmas
        self.x = None
        self.y = None
        
    @abstractmethod
    def generate_data(self, scorenet):
        pass
    
    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def make_extrapolation(self, x_pred):
        pass


class InterpolationExtrapolator:
    def __init__(self, method, data, x_mod):
        """
        Initialize an extrapolator object with an interpolation method, data, and x_mod.
        Parameters:
        method (str): The name of the interpolation method to use.
                      One of "linear", "quadratic", "cubic", or "spline".
        data (array-like): The data points to use for interpolation.
        x_mod (array-like): The x-values to extrapolate to.
        """
        self.method = method
        self.data = data
        self.x_mod = x_mod
        self.x = None
        self.y = None
        
    def generate_data(self):
        """
        Generate training data from the specified data points.
        """
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        
    def train_model(self):
        """
        Train the interpolation model using the generated training data.
        Raises a ValueError if no training data has been generated.
        """
        if self.x is None or self.y is None:
            raise ValueError("Training data not generated")
        if self.method == "linear":
            self.model = make_pipeline(LinearRegression())
        elif self.method == "quadratic":
            self.model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        elif self.method == "cubic":
            self.model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        elif self.method == "spline":
            self.model = CubicSpline(self.x, self.y, bc_type="natural")
        self.model.fit(self.x.reshape(-1, 1), self.y)
        
    def make_extrapolation(self, x_pred):
        """
        Use the trained interpolation model to make a prediction for the given x values.
        Returns:
        array-like: The predicted y values.
        Raises a ValueError if no training data has been generated.
        """
        if self.x is None or self.y is None:
            raise ValueError("Training data not generated")
        if self.method == "spline":
            return self.model(self.x_mod)
        else:
            return self.model.predict(self.x_mod.reshape(-1, 1))
        
        
class RegressionExtrapolator(Extrapolator):
    def __init__(self, model, data_shape, x_mod, sigmas):
        """
        Initialize an extrapolator object with a regression model, data shape, and sigmas.

        Parameters:
        model (object): The regression model to use for extrapolation.
        data_shape (list or tuple): The shape of the data points.
        x_mod (torch tensor): the x-value score model will be used on.
        sigmas (list): The standard deviations of diffusion (Gaussian) noise level.
        """
        super().__init__(data_shape, x_mod, sigmas)
        self.model = model
        self.sigmas = []
        self.labels = []
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(self.data_shape[0], device=x_mod.device) * c
            labels = labels.long()
            sigmas = torch.ones(self.data_shape, device=x_mod.device) * sigma
            self.labels += [labels]
            self.sigmas += [sigmas]
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
        self.x = np.array([torch.flatten(x).numpy() for x in self.sigmas])
        self.y = []
        sigma_min = self.sigmas[-1]
        for labels, sigma in zip(self.labels, self.sigmas):
            score = scorenet(self.x_mod, labels)
            # might need to standardize score to fit the model better?
            score *= (sigma / sigma_min) ** 2      # scaling by sigmas squares
            self.y += [score]
        self.y = np.array([torch.flatten(y).numpy() for y in self.y])
        
    def train_model(self):
        """
        Train the regression model using the generated training data.
        Raises a ValueError if no training data has been generated.
        """
        if self.x is None or self.y is None:
            raise ValueError("Training data not generated")
        self.model.fit(self.x.reshape(-1, 1), self.y)
        
    def make_extrapolation(self, x_pred):
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



# still need to modify this
@torch.no_grad()
def extrapolated_Langevin_dynamics(x_mod, scorenet, extrapolator, sigmas, n_steps=200, step_size=0.000008,
                             final_only=False, verbose=False, denoise=True):
    images = []
        for s in range(n_steps):
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
