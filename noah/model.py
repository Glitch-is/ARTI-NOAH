import tensorflow as tf
import numpy as np
import sys
import os
from tqdm import tqdm

class Model:

    # xlen: length of input sequence
    # ylen: length of output sequence
    # xsize: size of input vocabulary
    # ysize: size of output vocabulary
    # hidden_size: the hidden size of the LSTM
    # embedding_size: the size of the word representation within the RNN
    # num_layers: ... number of layers
    # save_path: the path to a checkpoint file saved during training
    # learning_rate: the learning rate
    # epochs: the number of iterations to learn
    def __init__(self, xlen, ylen, 
            xsize, ysize,
            hidden_size, embedding_size, num_layers, save_path,
            learning_rate=0.001, dropout_prob=0.5,
            epochs=100000, model_name='seq2seq_model', train=True):

        # attach these arguments to self
        self.xlen = xlen
        self.ylen = ylen
        self.save_path = save_path
        self.epochs = epochs
        self.model_name = model_name
        self.is_training = train
        os.makedirs(self.save_path, exist_ok=True)

        tf.logging.vlog(tf.logging.INFO, "Initializing Model...")

        # placeholders
        tf.reset_default_graph()

        with tf.name_scope('encoder'):
            #  encoder inputs : list of indices of length xlen
            self.encoder_inputs = [ tf.placeholder(shape=[None,], 
                            dtype=tf.int32, 
                            name='ei_{}'.format(t)) for t in range(xlen) ]

        with tf.name_scope('decoder'):
            #  labels that represent the real outputs
            self.labels = [ tf.placeholder(shape=[None,], dtype=tf.int32, name='labels') for t in range(ylen) ]

            #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
            self.decoder_inputs = [ tf.placeholder(shape=[None,], dtype=tf.int32, name='inputs') for t in range(ylen) ]
            # The loss weights
            self.loss_weights = [ tf.placeholder(shape=[None,], 
                                                 dtype=tf.float32,
                                                 name='weights') for t in range(ylen) ]


        # define the basic cell
        if self.is_training:
            basic_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True),
                output_keep_prob=dropout_prob)
        else:
            basic_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        
        # stack cells together : n layered model
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell] * num_layers, state_is_tuple=True)

        self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoder_inputs,
            self.decoder_inputs, 
            stacked_lstm,
            xsize, 
            ysize, 
            embedding_size,
            feed_previous=(not self.is_training)
        )

        if self.is_training:

            # weighted loss
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.outputs, self.labels, self.loss_weights, ysize)

            # optimizer
            self.opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        tf.logging.vlog(tf.logging.INFO, "Finished Initializing Model!")


    # get the feed dictionary
    def get_feed(self, X, Ydecoder, Ylabels, Yweights):
        # print("X:")
        # print(len(X))
        # print("Encoder in:")
        # print(len(self.encoder_inputs))
        # print("Xlen:")
        # print(self.xlen)

        feed = {self.encoder_inputs[t]: X[t] for t in range(self.xlen)}
        feed.update({self.labels[t]: Ylabels[t] for t in range(self.ylen)})
        feed.update({self.decoder_inputs[t]: Ydecoder[t] for t in range(self.ylen)})
        feed.update({self.loss_weights[t]: Yweights[t] for t in range(self.ylen)})
        return feed

    def get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        return tf.Session(config=config)

    def train(self, train_generator, sess=None, save_every=500):
        # save the model every time we advance by a percentage point
        saver = tf.train.Saver()

        # If the session parameter is not given, we construct a new one
        if not sess:
            # create a session
            sess = self.get_session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        tf.logging.vlog(tf.logging.INFO, "Start Training")
        # run M epochs
        step = 0
        for i in range(self.epochs):
            print("Epoch {}/{}".format(i, self.epochs))
            try:
                batch = train_generator.__next__()
                for vals in batch:
                    feed = self.get_feed(*vals)
                    _, loss = sess.run([self.opt_op, self.loss], feed)
                    step += 1
                    if step % 100 == 0:
                        tqdm.write("----- Step: %d ---- Loss: %.2f ------\n" % (step, loss))
                    if step % save_every == 0:
                        # save model to disk
                        saver.save(sess, self.save_path + self.model_name + '.ckpt', global_step=i)
                        # evaluate to get validation loss
                        # val_loss = self.eval_batches(sess, valid_set, 8)
                        # print stats
                        print('Model saved to disk at iteration #{}'.format(step))
                        # print('val loss : {0:.6f}'.format(val_loss))
                        sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                tf.logging.vlog(tf.logging.INFO, "Interrupted by user at iteration {}".format(i))
                break
            except StopIteration: # We don't have any more training data
                tf.logging.vlog(tf.logging.INFO, "Ran out of training data at iteration {}".format(i))
                break
        else:
            tf.logging.vlog(tf.logging.INFO, "Finished Training!")

        saver.save(sess, self.save_path + self.model_name + '.ckpt', global_step=i)
        # print stats
        print('Model saved to disk at end of training')
        sys.stdout.flush()
        self.session = sess
        return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = self.get_session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        assert(len(X) == self.args.xlen)
        feed = {self.enccoder_inputs[t]: X[t] for t in range(self.xlen)}

        dec_op_v = sess.run(self.outputs, feed)
        print(dec_op_v)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


