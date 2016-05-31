# Considering the amount of stuff I had to import from blocks all the time
# I found it more and more annoying to have to find and rerun a huge import cell
# every time in my notebook.
# So I'm trying to solve this by providing a "namespace" of sorts

import pickle
import logging

import numpy as np
import pandas as pd

# Theano stuff
import theano
from theano import tensor

# All sorts of bricks
from blocks.model import Model
from blocks.bricks import Linear
from blocks.bricks import NDimensionalSoftmax
from blocks.bricks import Rectifier
from blocks.bricks import Tanh
from blocks.bricks import Bias
from blocks.bricks import Initializable
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, BaseRecurrent, recurrent
from blocks.bricks.attention import AbstractAttention
from blocks.bricks.sequences import Sequence
from blocks.bricks.parallel import Fork, Merge
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.cost import MisclassificationRate
from blocks.bricks.sequence_generators import (SequenceGenerator,
                                               Readout,
                                               SoftmaxEmitter,
                                               LookupFeedback)

# All sorts of blocks
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale, Adam

# Fuel Imports
from fuel.datasets import IndexableDataset, Dataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, SequentialExampleScheme

from fuel.transformers import Padding, Mapping, Batch

# Main Loop
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks_extras.extensions.plot import Plot

from blocks.monitoring import aggregation

# (De)serialization
from blocks.serialization import dump, load

stream_log_handler = logging.StreamHandler()
