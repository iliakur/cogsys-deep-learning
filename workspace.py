# Considering the amount of stuff I had to import from blocks all the time
# I found it more and more annoying to have to find and rerun a huge import cell
# every time in my notebook.
# So I'm trying to solve this by providing a "namespace" of sorts


import numpy as np
import pandas as pd

# Theano stuff
from theano import tensor

# All sorts of bricks
from blocks.bricks import Linear
from blocks.bricks import NDimensionalSoftmax
from blocks.bricks import Rectifier
from blocks.bricks import Bias
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks.sequences import Sequence
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.bricks.cost import MisclassificationRate

# All sorts of blocks
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale, Adam

# Fuel Imports
from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, SequentialExampleScheme
from fuel.datasets import IndexableDataset

from fuel.transformers import Padding, Mapping, Batch

# Main Loop
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks_extras.extensions.plot import Plot
