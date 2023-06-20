import pickle
import time
from typing import Optional

def save_params(path: str, params: dict, show: bool = False, add_time: bool = True):
    """
    Save parameters to a file.

    Parameters
    ----------
    path : str
        Path to the file.

    params : dict
        Dictionary of parameters.

    show : bool, optional [default = False]
        Whether to print the parameters.

    add_time : bool, optional [default = True]
        Whether to add a timestamp to the file name.
    
    """
    if add_time:
        filepath = path + "_" + time.strftime("%Y%m%d-%H%M%S") + ".pkl"
    else:
        filepath = path + ".pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    if show:
        print("----------------------------------")
        print("Parameters saved to", filepath)
        for key, value in params.items():
            if hasattr(value, '__len__') and len(value) > 1000:
                print(key, " : ", "Too long to show")
            else:
                print(key, " : ", value)

def load_params(path: str, show: bool = False, global_vars: Optional[dict] = None):
    """
    Load parameters from a file.

    Parameters
    ----------
    path : str
        Path to the file.

    show : bool, optional [default = False]
        Whether to print the parameters.

    global_vars : dict, optional [default = None]
        Dictionary of global variables.

    Returns
    -------
    params : dict
        Dictionary of parameters.

    """
    with open(path, 'rb') as f:
        params = pickle.load(f)
    
    if show:
        print("----------------------------------")
        print("Parameters loaded from", path)
        for key, value in params.items():
            if hasattr(value, '__len__') and len(value) > 1000:
                print(key, " : ", "Too long to show")
            else:
                print(key, " : ", value)

    if global_vars is not None:
        for key, value in params.items():
            global_vars[key] = value

    return params
    