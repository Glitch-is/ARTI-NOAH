import noah.dataset
from noah.dataset import Dataset
from noah.model import Model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


dataset = Dataset("data/twitter.txt")
xlen = ylen = 25
xsize = ysize = len(dataset.word2id)
hidden_size = 512
embedding_size = 128
num_layers=3
save_path="save/twitter/"
batch_size=25
epochs=200
model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize,
              hidden_size=hidden_size,
              embedding_size=embedding_size,
              num_layers=num_layers,
              save_path=save_path,
              epochs=epochs)

training_batch = dataset.getRandomBatch(dataset.getTrainingData(), batch_size)
test_batch = dataset.getRandomBatch(dataset.getTestingData(), batch_size)
model.train(training_batch, test_batch)





