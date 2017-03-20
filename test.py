from noah.dataset import Dataset
from noah.model import Model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


dataset = Dataset("data/test/samples.txt")
xlen = ylen = 25
xsize = ysize = len(dataset.word2id)
hidden_size = 512
embedding_size = 128
num_layers=3
save_path="save/samples/"
batch_size=5
model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize, 
              hidden_size=hidden_size, 
              embedding_size=embedding_size, 
              num_layers=num_layers, 
              save_path=save_path)

session = model.restore_last_session()

input_batch = dataset.getRandomBatch(dataset.getTestingData(), batch_size).__next__()[0]
output = model.predict(session, input_batch)

for i, o in zip(input_batch.T, output):
    q = dataset.sequence2str(i[::-1])
    a = dataset.sequence2str(o)
    print("Q: {}\nA: {}".format(q, a))

