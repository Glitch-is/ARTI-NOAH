import tensorflow as tf
import numpy as np
import sys

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
        self.dropout_prob = dropout_prob
        self.is_training = train

        tf.logging.vlog(tf.logging.INFO, "Initializing Model...")

        # placeholders
        tf.reset_default_graph()

        #  encoder inputs : list of indices of length xlen
        self.encoder_inputs = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int32, 
                        name='ei_{}'.format(t)) for t in range(xlen) ]

        #  labels that represent the real outputs
        self.labels = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int32, 
                        name='el_{}'.format(t)) for t in range(ylen) ]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int32, name='GO') ] + self.labels[:-1]


        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)
        # define the basic cell
        basic_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
        # stack cells together : n layered model
        stacked_lstm = tf.contrib.rnn.MultiRNNCell([basic_cell] * num_layers, state_is_tuple=True)

        if self.is_training:
            # for parameter sharing between training model
            #  and testing model
            with tf.variable_scope('decoder') as scope:
                # build the seq2seq model 
                #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
                self.decode_outputs, self.decode_states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(self.encoder_inputs,self.decoder_inputs, stacked_lstm,
                                                    xsize, ysize, embedding_size)
                # share parameters
                scope.reuse_variables()
                # testing model, where output of previous timestep is fed as input 
                #  to the next timestep
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.encoder_inputs, self.decoder_inputs, stacked_lstm, xsize, ysize,embedding_size,
                    feed_previous=True)

            # weighted loss
            # the weights are initially all the same
            self.loss_weights = [ tf.placeholder(shape=[None,], 
                                                 dtype=tf.float32,
                                                 name='dl_{}'.format(t)) for t in range(ylen) ]
            self.loss = tf.contrib.legacy_seq2seq.sequence_loss(self.decode_outputs, self.labels, self.loss_weights, ysize)

            # train op to minimize the loss
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        else:
            with tf.variable_scope('decoder') as scope:
                self.decode_outputs_test, self.decode_states_test = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                    self.encoder_inputs, self.decoder_inputs, stacked_lstm, xsize, ysize,embedding_size,
                    feed_previous=True)
        tf.logging.vlog(tf.logging.INFO, "Finished Initializing Model!")


    # get the feed dictionary
    def get_feed(self, X, Y, Yw, keep_prob):
        # print("X:")
        # print(len(X))
        # print("Encoder in:")
        # print(len(self.encoder_inputs))
        # print("Xlen:")
        # print(self.xlen)

        feed = {self.encoder_inputs[t]: X[t] for t in range(self.xlen)}
        feed.update({self.labels[t]: Y[t] for t in range(self.ylen)})
        if self.is_training:
            feed.update({self.loss_weights[t]: Yw[t] for t in range(self.ylen)})
        # probability of a dropout
        feed[self.keep_prob] = keep_prob
        return feed

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY, weightY = train_batch_gen.__next__()
        # build feed
        feed = self.get_feed(batchX, batchY, weightY, keep_prob=self.dropout_prob)
        _, loss_v = sess.run([self.train_op, self.loss], feed)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY, weightY = eval_batch_gen.__next__()
        # build feed
        feed = self.get_feed(batchX, batchY, weightY, keep_prob=1.0)
        loss_v, dec_op_v = sess.run([self.loss, self.decode_outputs_test], feed)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        return loss_v, dec_op_v, batchX, batchY

    # evaluate 'num_batches' batches
    def eval_batches(self, sess, eval_batch_gen, num_batches):
        losses = []
        for i in range(num_batches):
            loss_v, dec_op_v, batchX, batchY = self.eval_step(sess, eval_batch_gen)
            losses.append(loss_v)
        return np.mean(losses)

    def train(self, train_set, valid_set, sess=None):
        if not self.is_training:
            raise ValueError("Training is disabled!")
        # save the model every time we advance by a percentage point
        saver = tf.train.Saver()

        # If the session parameter is not given, we construct a new one
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())

        tf.logging.vlog(tf.logging.INFO, "Start Training")
        # run M epochs
        for i in range(self.epochs):
            try:
                loss_v = self.train_batch(sess, train_set)
                if i and i % (self.epochs // 100) == 0: 
                    # save model to disk
                    saver.save(sess, self.save_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    # val_loss = self.eval_batches(sess, valid_set, 8)
                    # print stats
                    print('Model saved to disk at iteration #{}'.format(i))
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
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        assert(len(X) == self.xlen)
        feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.xlen)}
        feed_dict[self.keep_prob] = 1.0
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        print(dec_op_v)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


