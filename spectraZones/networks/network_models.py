import torch
from torch.utils.data import DataLoader
import random
import numpy as np

from matplotlib import pyplot as plt
import pandas as pd

from matplotlib import pyplot as plt

import os
from tqdm.notebook import tqdm


plt.style.use("fivethirtyeight")

from IPython.display import set_matplotlib_formats

set_matplotlib_formats("pdf", "png")
pd.options.display.float_format = "{:.5f}".format

rc = {
    "savefig.dpi": 350,
    "figure.autolayout": False,
    "figure.figsize": [15, 5],
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "font.size": 8,
    "lines.linewidth": 2.0,
    "lines.markersize": 8,
    "legend.fontsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
}

import seaborn as sns

sns.set_theme(style="darkgrid", rc=rc)


from .network_bodies import *
from .network_utils import *


class BaseModel:
    """
    Base NN Model

    """

    def __init__(self, config=Config(), work_dir="", seed=420):
        self.work_dir = work_dir
        self.seed = seed
        self.setSeed()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_file = "checkpoint.pt"

        self.config = config
        self.criterion = config.criterion

        self.train_epoch = 0
        self.valid_loss_min = np.Inf
        self.losses = []
        self.val_losses = []

        txt = f"Starting NN Model at device: {self.device}"
        print()
        print("-" * len(txt))
        print(f"Starting NN Model at device: {self.device}")
        print("-" * len(txt))
        print()

    @property
    def save_path(self):
        return os.path.join(self.work_dir, self.model_file)

    def setSeed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def tensor(self, np_arr, device=None):
        if device is None:
            device = self.device
        #         print(self.device)
        return torch.tensor(np_arr, dtype=torch.float64, device=device)

    def toNumpy(self, tensor):
        return tensor.cpu().detach().numpy()

    def load_weights(self, file):
        self.nn.load_state_dict(torch.load(file))

    def saveModel(self):
        """
        Save Model
        """

        torch.save(
            {
                "epoch": self.train_epoch,
                "model_state_dict": self.nn.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.losses,
                "val_loss": self.val_losses,
                "config": self.config,
            },
            self.save_path,
        )

    def loadModel(self):
        """
        Load Model
        """

        checkpoint = torch.load(self.save_path, weights_only=False)
        self.nn.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_epoch = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]

    @classmethod
    def loadModelFile(
        cls,
        model_name="autoencolder",
        save_path="models/",
    ):
        """
        Load Model

        Parameters
        ----------
        model_name : str
            Model name to load.
        save_path : str
            Model save directory.
        """

        checkpoint_path = os.path.join(save_path, f"{model_name}.pt")

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model = cls(
            config=checkpoint["config"], model_name=model_name, save_path=save_path
        )
        model.loadModel()
        return model

    def plot_scores(self, train, val=None, rolling_window=10, title=""):
        """
        Plot Agent Traininig Scores per Episode.

        Params
        ======
            scores (list): Scores over each episode.
            rolling_window (int): Rolling mean window length.
            title (string): Chart title.
        """
        fig = plt.figure(dpi=127)
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(train)), train, label="Train", lw=0.3)
        if val is not None:
            plt.plot(np.arange(len(val)), val, label="Val", lw=0.3)
        plt.ylabel("Score")
        plt.xlabel("Epoch #")
        #     rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        #     plt.plot(rolling_mean)
        plt.title(title)
        plt.legend()
        plt.show()


class AutoEncoderModel(BaseModel):
    def __init__(
        self,
        config,
        model_name="autoencolder",
        save_path="models/",
        random_seed=420,
    ):
        """
        Deep AutoEncoder Model

        Parameters
        ----------
        config : Config
            Model configuration object.
            Model parameters:
                - input_dim: int
                    DAE input dimension.
                - start_filter: int
                    Number of filters in the first layer.
                - depth: int
                    Number of layers.
                - criterion: torch.nn
                    Loss function for training the model.
                - optimizer: torch.optim
                    Optimizer for training the model.
                - lr: float
                    Learning rate.
                - lr_w_decay: float
                    Weight decay for the optimizer.
                - batch_size: int
                    Batch size for training the model.
                - lr_scheduler: torch.optim.lr_scheduler
                    Learning rate scheduler.
                - lr_patience: int
                    Learning rate scheduler patience.
                - lr_delta: float
                    Learning rate scheduler delta.
                - lr_decay_factor: float
                    Learning rate scheduler decay factor.
                - es_scheduler: torch.optim.lr_scheduler
                    Early stopping scheduler.
                - es_patience: int
                    Early stopping patience.
                - es_delta: float
                    Early stopping threshold.

        model_name : str
            Model name (also used to name the model file).
        save_path : str
            Directory to save the model.
        random_seed : int
            Random seed for the model.
        """
        super().__init__(config, seed=random_seed)

        self.nn = AutoencoderNN(config.input_dim, config.start_filter, config.depth)
        self.criterion = config.criterion
        self.optimizer = config.optimizer(
            self.nn.parameters(), lr=config.lr, weight_decay=config.lr_w_decay
        )
        self.lr_scheduler = (
            config.lr_scheduler(
                self.optimizer,
                patience=config.lr_patience,
                threshold=config.lr_delta,
                factor=config.lr_decay_factor,
            )
            if config.lr_scheduler
            else False
        )
        self.es_scheduler = (
            config.es_scheduler(
                patience=config.es_patience, verbose=True, delta=config.es_delta
            )
            if config.es_scheduler
            else False
        )

        self.work_dir = save_path
        self.model_name = model_name
        self.model_file = f"{model_name}.pt"
        self.train_epoch = -1

    def setData(self, X):
        # Check inputs
        assert len(X.shape) == 2, "Input must be 2D numpy array with"
        return DataLoader(X, batch_size=self.config.batch_size, shuffle=True)

    def train(self, X, epochs=500, print_each=20):
        print(f"Start Training Model: {self.model_name}")
        print(f"  - Total Epochs: {epochs}")

        self.setSeed()

        train_loader = self.setData(X)
        self.losses = []

        for epoch in (epoch_pb := tqdm(range(epochs), desc="Training Epoch")):
            self.train_epoch = epoch
            epoch_losses = []
            for data, i in zip(
                train_loader,
                (
                    bach_pb := tqdm(
                        range(len(train_loader)), leave=False, desc="Training Batch"
                    )
                ),
            ):
                input_ = data.float()
                # ===================forward=====================
                self.nn.train()
                self.nn.zero_grad()

                output = self.nn(input_)
                loss = self.criterion(output, input_)
                epoch_losses.append(loss.item())

                if (i + 1) % print_each == 0 or i == 0:
                    bach_pb.set_description(
                        f"Epoch {epoch} | Batch Loss: {np.mean(epoch_losses):.8f} | ",
                        refresh=True,
                    )
                # ===================backward====================
                loss.backward()
                self.optimizer.step()
            # ===================log========================
            #             self.timer.update(epoch+1)
            mean_loss = np.mean(epoch_losses)
            self.losses.append(mean_loss)
            epoch_pb.set_description(f"Epoch Loss: {mean_loss:.8f} | ", refresh=True)

            if self.es_scheduler:
                self.es_scheduler(mean_loss, mean_loss, self)
                if self.es_scheduler.early_stop:
                    break

            if self.lr_scheduler:
                self.lr_scheduler.step(mean_loss)

        self.saveModel()
        self.plot_scores(self.losses, rolling_window=10, title="Autoencolder Training")

    def encode(self, data):
        """
        Encodes the input data using the trained encoder of the autoencoder model.

        Parameters
        ----------
        data : numpy array
            Input data to encode (n_samples, n_bands).

        """
        assert len(data.shape) == 2, "Input must be 2D numpy array with"
        assert (
            data.shape[1] == self.config.input_dim
        ), f"Input shape diferent ({data.shape[1]}) from training shape ({self.config.input_dim})"

        encoded = self.nn.encoder(self.tensor(data, device="cpu").float())
        return self.toNumpy(encoded)

    def decode(self, data):
        """
        Decodes the input data using the trained decoder of the autoencoder model.

        """
        assert len(data.shape) == 2, "Input must be 2D numpy array with"
        assert (
            data.shape[1] == self.nn.encoder[-1].out_features
        ), f"Input shape different ({data.shape[1]}) from training shape ({self.config.input_dim})"

        decoded = self.nn.decoder(self.tensor(data, device="cpu").float())
        return self.toNumpy(decoded)
