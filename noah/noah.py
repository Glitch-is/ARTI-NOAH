import argparse
import tensorflow as tf
import numpy as np
import math

from tensorflow.python import debug as tf_debug

from noah.dataset import DataSet
from noah.model import Model

class Noah:
    def __init__(self):
        pass

    def main(self):
        print("Initializing NOAH: Neural Oriented Aritificial Human...")

        tf.logging.set_verbosity(tf.logging.DEBUG) # DEBUG, INFO, WARN (default), ERROR, or FATAL

        # Fetch and prepare text
        self.dataset = DataSet()

        # pass the data to our model
        self.model = Model(self.dataset)

        # Start a tensorflow session
        self.sess = tf.Session()
