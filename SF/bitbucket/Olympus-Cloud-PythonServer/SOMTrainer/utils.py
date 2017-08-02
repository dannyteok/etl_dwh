import os
import numpy as np


def file_exists(configuration, suffix):
    path = get_path(configuration=configuration,suffix=suffix)

    return False


def get_path(configuration, suffix=None):
   if suffix:
       return os.path.join(configuration.get('data_path',"/sites"),
                               configuration['owner'],
                               configuration['site'],
                               configuration['floor'],
                                suffix
                               )
   else:
       return os.path.join(configuration.get("data_path","/sites"),
                        configuration['owner'],
                        configuration['site'],
                        configuration['floor'],
                        )

def unique_rows(input_array):
    b = np.ascontiguousarray(input_array).view(np.dtype((np.void, input_array.dtype.itemsize * input_array.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = input_array[idx]
    return unique_a
