import noah.dataset
from noah.dataset import Dataset
from noah.model import Model


dataset = Dataset("data/test/test.txt")
xlen = ylen = 25
xsize = ysize = len(dataset.word2id)
hidden_size = 512
embedding_size = 128
num_layers=3
save_path="save/test/"
print(dataset.trainingExamples)
print(dataset.testingExamples)
model = Model(xlen=xlen, ylen=ylen, xsize=xsize, ysize=ysize, 
              hidden_size=hidden_size, 
              embedding_size=embedding_size, 
              num_layers=num_layers, 
              save_path=save_path)

model.train(dataset.trainingExamples.__iter__(), dataset.testingExamples.__iter__())





