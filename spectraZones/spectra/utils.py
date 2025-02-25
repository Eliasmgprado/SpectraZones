from pysptools import spectro
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import multiprocessing
import traceback


def remove_continuum(a, wvl_arr):
    assert isinstance(wvl_arr, np.ndarray) or isinstance(
        wvl_arr, list
    ), "Error, wvl_arr must be a list or a numpy array."
    if isinstance(wvl_arr, list):
        wvl_arr = np.array(wvl_arr)

    if len(wvl_arr.shape) == 1:
        wvl_arr = wvl_arr.reshape((-1, 1))
    assert (
        len(wvl_arr.shape) == 2 and wvl_arr.shape[1] == 1
    ), "Error, wvl_arr must be a 1D array, or a 2D array with shape (n, 1)."

    return spectro.convex_hull_removal(a, wvl_arr.astype(float))[0]


def remove_continuum_aux(arr, wvl):
    try:
        return np.array(remove_continuum(np.array(arr), wvl)).reshape((-1, 1))
    except Exception:
        traceback.print_exc()
        return None


def remove_continuum_parallel(input_data, wvl, num_cores=None):
    """
    Remove convex hull from all spectra in input_data
    using parallel processing.

    Parameters
    ----------
    input_data : numpy array
        Array of spectra to process (n_samples, n_bands).
    wvl : numpy array
        Array of wavelengths (n_bands).
        
    Returns
    -------
    numpy array
        Array of continuum removed spectra (n_samples, n_bands).
    """

    # Determine number of CPU cores; you can also set n_jobs manually.
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} cores for parallel processing.")

    # Parallel execution using joblib; results are returned in the same order as all_swir.
    results = Parallel(n_jobs=num_cores)(
        delayed(remove_continuum_aux)(sample, wvl)
        for sample in tqdm(input_data, desc="Processing samples", total=len(input_data))
    )

    # Combine results into a numpy array.
    return np.array(results)
