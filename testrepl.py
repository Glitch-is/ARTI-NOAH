from load import main
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)


dataset, model = main(train=False)

session = model.restore_last_session()

while True:
    line = input("Q: ")
    line = line.strip()
    if not line:
        continue
    inp = dataset.str2sequence(line)
    print(inp)
    output = model.predict(session, inp.T)
    q = dataset.sequence2str(inp[0][::-1])
    a = dataset.sequence2str(output[0])
    print("Q: {}\nA: {}".format(q, a))


