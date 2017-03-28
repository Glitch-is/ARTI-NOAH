from load import main
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

batch_size=256
dataset, model = main(train=True)

training = dataset.getTrainingData()
training_batch = dataset.getBatches(training, batch_size)
test_batch = dataset.getRandomBatch(dataset.getTestingData(), batch_size)
model.train(training_batch, test_batch)
