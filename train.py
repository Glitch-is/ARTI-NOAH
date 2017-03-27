import noah.dataset
from noah.dataset import Dataset
from noah.model2 import Model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


xlen = ylen = 5
dataset = Dataset("data/twitter.txt", maxX=xlen, maxY=ylen)
xsize = ysize = len(dataset.word2id)
hidden_size = 512
embedding_size = 25
num_layers=3
save_path="save/twitter/"
batch_size=40
epochs=1000
learning_rate=0.002
model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize,
              hidden_size=hidden_size,
              embedding_size=embedding_size,
              num_layers=num_layers,
              save_path=save_path,
              epochs=epochs,
              learning_rate=learning_rate,
              model_name="blagh2",
              train=True)

training = dataset.getTrainingData()
training_batch = dataset.getRandomBatch(training, batch_size)
test_batch = dataset.getRandomBatch(dataset.getTestingData(), batch_size)
model.train(training_batch, test_batch)





