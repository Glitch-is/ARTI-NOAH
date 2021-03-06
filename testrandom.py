from load import main
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


dataset, model = main(train=False)



input_batch, _, output_batch, _ = dataset.getRandomBatch(dataset.getTrainingData(), 1).__next__()
for i, o in zip(input_batch.T, output_batch.T):
    print(i, o)
    q = dataset.sequence2str(i[::-1])
    a = dataset.sequence2str(o)
    print("Q: {}\nA: {}".format(q, a))

session = model.restore_last_session()
output = model.predict(session, input_batch)
for i, o in zip(input_batch.T, output):
    print(list(i), o)
    q = dataset.sequence2str(i[::-1])
    a = dataset.sequence2str(o)
    print("Q: {}\nA: {}".format(q, a))


