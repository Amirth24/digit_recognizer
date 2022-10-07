import numpy as np
def dict_to_ndarray(data, width, height, channels = 3):
    return np.array([int(x) for _, x in data.items()]).reshape(width, height, channels)