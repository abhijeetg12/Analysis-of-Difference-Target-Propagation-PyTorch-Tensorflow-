'''
Target propagation for a 2-layer network
    layer:     [  0  ]    [          1           ]     [   3  ]    [   top    ]
    network:   [input] -> [affine -> nonlinearity]  -> [affine] -> [loss layer]
'''

import os
import numpy as np
import targprop.datasets as ds
import targprop.operations as ops
from scipy import linalg
import tensorflow as tf