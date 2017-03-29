from noah.dataset import Dataset
from noah.model import Model


def main(train=True, model_name="blagh"):
    xlen = 5
    ylen = xlen + 2
    dataset = Dataset("data/cornell/", maxX=xlen, maxY=ylen, corpus="cornell")
    xsize = ysize = len(dataset.word2id)
    hidden_size = 512
    embedding_size = 30
    num_layers=2
    save_path="save/cornell3/"
    epochs=50
    learning_rate=0.002
    dropout_prob=0.9

    model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize,
                  hidden_size=hidden_size,
                  embedding_size=embedding_size,
                  num_layers=num_layers,
                  save_path=save_path,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  model_name=model_name,
                  train=train,
                  dropout_prob=dropout_prob)
    return dataset, model
