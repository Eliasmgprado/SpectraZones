import os
import pickle
from scipy import optimize
import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm.notebook import tqdm


def rescale_stanley(arr, eps=1e-3):
    """
    Rescale the input array to the range [eps, 1 - eps].

    Parameters
    ----------
    arr : np.ndarray
        Input array (n_samples,).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    np.ndarray
        Rescaled array (n_samples,).
    """
    return (((arr - arr.min()) / (arr.max() - arr.min())) * (1 - 2 * eps)) + eps


def rescale_stanley_torch(tensor, eps=1e-3):
    """
    Rescale the input tensor to the range [eps, 1 - eps].

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (n_samples,).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Rescaled tensor (n_samples,).
    """
    min_vals = torch.min(tensor, dim=0, keepdim=True)[0]
    max_vals = torch.max(tensor, dim=0, keepdim=True)[0]
    return (((tensor - min_vals) / (max_vals - min_vals)) * (1 - 2 * eps)) + eps


def power_transform(arr, lamb):
    """
    Applies the power transformation to the input array using the specified lambda.

    Parameters
    ----------
    arr : np.ndarray
        Input array (n_samples,).
    lamb : float
        The power parameter for transformation.

    Returns
    -------
    np.ndarray
        Transformed array (n_samples,).
    """
    return np.power(arr, lamb)


def power_transform_std(lamb, arr):
    """
    Applies the power transformation to the input array using the specified lambda 
    and returns the negative standard deviation.

    Parameters
    ----------
    lamb : np.ndarray
        The power parameter for transformation.
    arr : np.ndarray
        Input array (n_samples,).

    Returns
    -------
    float
        Negative standard deviation of the transformed array.
    """
    return -np.power(arr, lamb[0]).std()


def find_lamb(arr, from_=0.01, to_=100, step=0.01):
    """
    Find the best lambda for power transformation using brute force optimization.

    Parameters
    ----------
    arr : np.ndarray
        Input array (n_samples,).
    from_ : float
        Start of the search space.
    to_ : float
        End of the search space.
    step : float
        Step size for the search space.

    Returns
    -------
    float
        Best lambda for power transformation.
    """

    r_vals = rescale_stanley(arr)

    resbrute = optimize.brute(
        power_transform_std,
        (slice(from_, to_, step),),
        args=(r_vals,),
        full_output=True,
        finish=optimize.fmin,
    )
    best_lamb = resbrute[0][0]
    print(f"original std: {r_vals.std()}")
    print(f"best lamb: {best_lamb}")
    print(f"max std: {-resbrute[1]}")

    return best_lamb


class MDVPTTransform:
    """
    Maximum Data Variation Power Transform (MDVPT)

    MDVPT is a power transform that maximizes the data variation in a dataset.
    """

    def __init__(self, name="MDVPT", output_dir=""):
        """
        Parameters
        ----------
        name : str
            Name of the file (used as the name of the saved file).
        output_dir : str
            Directory to save the transformation model.
        """

        self.model_path = os.path.join(output_dir, f"{name}.pkl")
        self.lamb_arr = []

    def fit(self, arr, num_jobs=-1):
        """

        Fits the MDVPTTransform to the data by finding the best lambda for power transformation.

        Parameters
        ----------
        arr : np.ndarray
            Input array (n_samples, n_features).
        num_jobs : int
            Number of jobs to run in parallel; -1 means using all processors.
        """
        num_cols = arr.shape[1]

        # Use joblib to run find_lamb on each column in parallel.
        self.lamb_arr = Parallel(n_jobs=num_jobs)(
            delayed(find_lamb)(arr[:, i])
            for i in tqdm(range(num_cols), desc="Processing samples", total=num_cols)
        )

        self.lamb_arr = np.array(self.lamb_arr)

    def transform(self, arr, device=None):
        """
        Transforms the input array using the fitted MDVPT.

        Parameters
        ----------
        arr : np.ndarray
            Input array (n_samples, n_features).
        device : str
            Device to use for transformation; if None, uses cuda if available, else cpu.

        Returns
        -------
        np.ndarray
            Transformed array (n_samples, n_features).

        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            device = torch.device("cuda")
            arr_tensor = torch.tensor(arr, dtype=torch.float32, device=device)
            # Normalize using a vectorized PyTorch function.
            r_vals = rescale_stanley_torch(arr_tensor)
            # Create a tensor for lamb_arr and unsqueeze to enable broadcasting.
            lamb_tensor = torch.tensor(
                self.lamb_arr, dtype=torch.float32, device=device
            )
            transformed = torch.pow(r_vals, lamb_tensor.unsqueeze(0))
            return transformed.cpu().numpy()
        else:
            mdvpt_arr = np.zeros(arr.shape)
            for i in range(arr.shape[1]):
                r_vals = rescale_stanley(arr[:, i])
                mdvpt_arr[:, i] = np.power(r_vals, self.lamb_arr[i])
            return mdvpt_arr

    def fit_transform(self, arr, save=True, device=None):
        """
        Fits the MDVPT to the data and transforms the input array.

        Parameters
        ----------
        arr : np.ndarray
            Input array (n_samples, n_features).
        save : bool
            Whether to save the transformation model.
        device : str
            Device to use for transformation; if None, uses cuda if available, else cpu.

        """
        self.fit(arr)
        transformed = self.transform(arr, device=device)
        if save:
            self.save()
        return transformed

    def save(self):
        """
        Saves the transformation model to a file."""
        with open(self.model_path, "wb") as f:
            pickle.dump(self.lamb_arr, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        """
        Loads the transformation model from a file.
        """
        with open(self.model_path, "rb") as f:
            self.lamb_arr = pickle.load(f)
