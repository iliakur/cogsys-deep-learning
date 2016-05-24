__author__ = 'koller'


from scipy.io import loadmat
from blocks.bricks.conv import Convolutional, MaxPooling
from blocks.initialization import Constant
from blocks.bricks import Rectifier, Softmax
import numpy as np
import theano
from theano import tensor
from blocks.bricks.interfaces import Feedforward
from blocks.bricks.base import application
from blocks.bricks import *
from blocks.model import *
from blocks.graph import *
from blocks.bricks.conv import Convolutional, MaxPooling
from blocks.initialization import Constant
from blocks.bricks import Rectifier, Softmax
import theano.tensor as tt
import h5py
from blocks.bricks.conv import Flattener
from blocks.bricks.interfaces import Feedforward
from blocks.bricks.base import application

class ImagenetModel:
    """Loads a CNN in Matlab format and converts it to Blocks.
    Pass the filename of the Matlab file as an argument when you
    construct the ImagenetModel object 'im'. You can then retrieve
    a list of bricks from 'im.layers', and you can retrieve the
    model metadata (e.g. class labels) from 'im.M'."""

    def __init__(self, filename, imagesize=(256,256)):
        self.imagesize = imagesize

        print("Loading model ...")
        (L,M) = self.load_model(filename)

        self.M = M

        print("Converting model ...")
        self.layers = self.convert_model(L)


    def load_model(self, filename):
        """Loads a Matlab model into a Numpy array. Returns
        a tuple (L,M), where L contains the layers of the model
        and M contains the metadata (e.g. class labels)."""

        x = loadmat(filename, struct_as_record=True)
        return (x["layers"], x["meta"])

    def convert_model(self, L):
        """Converts a Matlab model into Blocks. The method expects
        an argument L specifying the layers of the Matlab model,
        e.g. as returned by load_model. It returns a list of bricks.
        This list may be longer than the list of layers in L, because
        additional padding bricks are introduced to work around
        limitiations in the Blocks pooling bricks."""

        layers = []
        image_size = self.imagesize

        for i in range(37):   # 37 layers in Matlab model
            l = L[0][i][0][0]
            tp = l["type"][0]
            name = l["name"][0]

            if tp == "conv":
                wt = l["weights"][0,0]
                bias = l["weights"][0,1]
                pad = l["pad"][0]
                stride = l["stride"][0]

                # WORK-AROUND to get to 7x7 output after last convolution
                if name == 'conv5_3':
                    pad = [0, 1, 0, 1]

                if sum(pad) > 0:
                    pad = [int(d) for d in pad]
                    layer = Padding(pad)
                    layer.image_size = image_size
                    image_size = layer.get_dim("output")[1:3]
                    layers.append(layer)

                layer, outdim = self.conv_layer(name, wt, bias, image_size)
                layers.append(layer)
                image_size = outdim

            elif tp == "pool":
                method = l["method"][0]
                pool = l["pool"][0]
                stride = l["stride"][0]
                pad = l["pad"][0]

                stride = [int(d) for d in stride]
                pool = [int(d) for d in pool]
                pad = [int(d) for d in pad]

                layer, outdim = self.pool_layer(name, method, pool, pad, stride, image_size)
                layers.append(layer)
                image_size = outdim

            elif tp == "relu":
                layers.append(self.relu_layer(name))

            elif tp == "softmax":
                layers.append(Flattener('flatten'))
                layers.append(self.softmax_layer(name))

        print(len(layers), 'layers created')
        return layers


    def to_bc01(self, shape_01cb):
        """Converts filters from the 01cb shape used by Matlab to the
        bc01 shape used by Blocks."""

        b01c = np.rollaxis(shape_01cb,3,0)
        return np.rollaxis(b01c,3,1)


    def conv_layer(self, name, wt, bias, image_size):
        """Creates a Convolutional brick with the given name, weights,
        bias, and image_size."""

        layer = Convolutional(name=name,
                              filter_size=wt.shape[0:2],
                              num_channels=wt.shape[2], # in
                              num_filters=wt.shape[3], # out
                              weights_init=Constant(0), # does not matter
                              biases_init=Constant(0), # does not matter
                              tied_biases=True,
                              border_mode='valid',
                            )

        if image_size:
            layer.image_size = image_size

        layer.initialize()

        weights = self.to_bc01(wt)
        layer.parameters[0].set_value(weights.astype("float32")) # W
        layer.parameters[1].set_value(bias.squeeze().astype("float32")) # b

        return (layer, layer.get_dim("output")[1:3])

    def pool_layer(self, name, method, pool, pad, stride, image_size):
        """Creates a MaxPooling brick with the given name, pooling size, stride,
        and image size. If a string other than 'max' is passed in the 'method'
        parameter, the function throws an exception. The 'pad'  argument
        are ignored. It is instead handled in the conversion through a Padding
        brick (see below)."""

        # FIX: ignore padding [0 1 0 1]

        if method == 'max':
            layer = MaxPooling(name=name, pooling_size=pool, step=stride, input_dim=image_size)
        else:
            raise Exception("Unsupported pooling method: %s" % method)

        return (layer, layer.get_dim("output"))

    def relu_layer(self, name):
        """Creates a Rectifier brick with the given name."""

        return Rectifier(name=name)

    def softmax_layer(self, name):
        """Creates a Softmax brick with the given name."""

        return Softmax(name=name)



class Padding(Feedforward):
        def __init__(self, pad=(1,1,1,1), value=0, **kwargs):
            super(Padding, self).__init__(**kwargs)
            self.pad = pad
            self.value = value

        def get_dim(self, name):
            if name == 'input_':
                return (None,) + self.image_size
            if name == 'output':
                return (None,self.image_size[0] + self.pad[0] + self.pad[1],
                        self.image_size[1] + self.pad[2] + self.pad[3])

        @application(inputs=['input_'], outputs=['output'])
        def apply(self, input_):
            shape = (input_.shape[0],
                    input_.shape[1],
                    input_.shape[2] + self.pad[0] + self.pad[1],
                    input_.shape[3] + self.pad[2] + self.pad[3])

            output = tt.alloc(float(self.value), *shape)

            output = tt.set_subtensor(output[:,:,
                                                self.pad[0]:-self.pad[1],
                                                self.pad[2]:-self.pad[3]],
                                        input_)

            return output

