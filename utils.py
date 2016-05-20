from workspace import *


def transpose_stream(data):
    #     data is a tuple, since it's expected to come from the padding transformer
    return tuple(np.swapaxes(item, 0, 1) for item in data)
