from noah.dataset import Dataset
from noah.model import Model

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)



xlen = ylen = 5
dataset = Dataset("data/opensubs/OpenSubtitles", maxX=xlen, maxY=ylen, corpus="opensubs")
xsize = ysize = len(dataset.word2id)
hidden_size = 512
embedding_size = 25
num_layers=3
save_path="save/opensubs/"
batch_size=256
epochs=30
learning_rate=0.002
model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize,
              hidden_size=hidden_size,
              embedding_size=embedding_size,
              num_layers=num_layers,
              save_path=save_path,
              epochs=epochs,
              learning_rate=learning_rate,
              model_name="blagh",
              train=False)

session = model.restore_last_session()

input_batch = dataset.getRandomBatch(dataset.getTestingData(), 1).__next__()[0]
output = model.predict(session, input_batch)


for i, o in zip(input_batch.T, output):
    print(list(i), o)
    q = dataset.sequence2str(i[::-1])
    a = dataset.sequence2str(o)
    print("Q: {}\nA: {}".format(q, a))


