import tensorflow as tf

class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        tf.logging.vlog(tf.logging.INFO, "Initializing Model...")
        tf.logging.vlog(tf.logging.INFO, "Finished Initializing Model!")
