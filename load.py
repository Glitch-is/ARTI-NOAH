from noah.dataset import Dataset
from noah.model import Model
import sys
import argparse

def main(*arg, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs="?", choices=["default", "cornell8", "cornell5", "twitter5", "ubuntu5"], const="default", default="default")
    args = parser.parse_args(sys.argv[1:])
    if args.run == "default":
        return default(*arg, **kwargs)
    elif args.run == "cornell8":
        return cornell8(*arg, **kwargs)
    elif args.run == "cornell5":
        return cornell5(*arg, **kwargs)
    elif args.run == "twitter5":
        return twitter5(*arg, **kwargs)
    elif args.run == "ubuntu5":
        return ubuntu5(*arg, **kwargs)

def default(train=True, model_name="blagh"):
    xlen = 5
    ylen = xlen + 2
    dataset = Dataset("data/cornell", maxX=xlen, maxY=ylen, corpus="cornell", clean=False)
    xsize = ysize = len(dataset.word2id)
    hidden_size = 512
    embedding_size = 30
    num_layers=2
    save_path="save/bla/"
    epochs=30
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

def cornell8(train=True, model_name="blagh"):
    xlen = 8
    ylen = xlen + 2
    dataset = Dataset("data/cornell", maxX=xlen, maxY=ylen, corpus="cornell", clean=False)
    xsize = ysize = len(dataset.word2id)
    hidden_size = 512
    embedding_size = 25
    num_layers=2
    save_path="save/cornell8/"
    epochs=30
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

def ubuntu5(train=True, model_name="blagh"):
    xlen = 5
    ylen = xlen + 2
    dataset = Dataset("data/ubuntu", maxX=xlen, maxY=ylen, corpus="ubuntu", clean=False)
    xsize = ysize = len(dataset.word2id)
    hidden_size = 512
    embedding_size = 30
    num_layers=2
    save_path="save/ubuntu/"
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

def twitter5(train=True, model_name="blagh"):
    xlen = 5
    ylen = xlen + 2
    dataset = Dataset("data/twitter/twitter_en.txt", maxX=xlen, maxY=ylen, corpus="txt", clean=False)
    xsize = ysize = len(dataset.word2id)
    hidden_size = 512
    embedding_size = 30
    num_layers=2
    save_path="save/twitter2/"
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

def cornell5(train=True, model_name="blagh"):
    xlen = 5
    ylen = xlen + 2
    dataset = Dataset("data/cornell", maxX=xlen, maxY=ylen, corpus="cornell", clean=False)
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
