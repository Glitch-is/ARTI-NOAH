import tensorflow as tf
import numpy as np
import sys

class Model:

    # xseq_len: length of input sequence
    # yseq_len: length of output sequence
    # xvocab_size: size of input vocabulary
    # yvocab_size: size of output vocabulary
    # hidden_size: the hidden size of the LSTM
    # embedding_size: the size of the word representation within the RNN
    # num_layers: ... number of layers
    # ckpt_path: the path to a checkpoint file saved during training
    # lr: the learning rate
    # epochs: the number of iterations to learn
    def __init__(self, xseq_len, yseq_len, 
            xvocab_size, yvocab_size,
            hidden_size, embedding_size, num_layers, ckpt_path,
            lr=0.0001, 
            epochs=100000, model_name='seq2seq_model'):

        # attach these arguments to self
        self.xseq_len = xseq_len
        self.yseq_len = yseq_len
        self.ckpt_path = ckpt_path
        self.epochs = epochs
        self.model_name = model_name

        tf.logging.vlog(tf.logging.INFO, "Initializing Model...")

        # placeholders
        tf.reset_default_graph()

        #  encoder inputs : list of indices of length xseq_len
        self.encoder_inputs = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int64, 
                        name='ei_{}'.format(t)) for t in range(xseq_len) ]

        #  labels that represent the real outputs
        self.labels = [ tf.placeholder(shape=[None,], 
                        dtype=tf.int64, 
                        name='el_{}'.format(t)) for t in range(yseq_len) ]

        #  decoder inputs : 'GO' + [ y1, y2, ... y_t-1 ]
        self.decoder_inputs = [ tf.zeros_like(self.encoder_inputs[0], dtype=tf.int64, name='GO') ] + self.labels[:-1]


        # Basic LSTM cell wrapped in Dropout Wrapper
        self.keep_prob = tf.placeholder(tf.float32)
        # define the basic cell
        basic_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True),
                output_keep_prob=self.keep_prob)
        # stack cells together : n layered model
        stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([basic_cell]*num_layers, state_is_tuple=True)


        # for parameter sharing between training model
        #  and testing model
        with tf.variable_scope('decoder') as scope:
            # build the seq2seq model 
            #  inputs : encoder, decoder inputs, LSTM cell type, vocabulary sizes, embedding dimensions
            self.decode_outputs, self.decode_states = tf.nn.seq2seq.embedding_rnn_seq2seq(self.encoder_inputs,self.decoder_inputs, stacked_lstm,
                                                xvocab_size, yvocab_size, embedding_size)
            # share parameters
            scope.reuse_variables()
            # testing model, where output of previous timestep is fed as input 
            #  to the next timestep
            self.decode_outputs_test, self.decode_states_test = tf.nn.seq2seq.embedding_rnn_seq2seq(
                self.encoder_inputs, self.decoder_inputs, stacked_lstm, xvocab_size, yvocab_size,embedding_size,
                feed_previous=True)

        # weighted loss
        # the weights are initially all the same
        loss_weights = [ tf.ones_like(label, dtype=tf.float32) for label in self.labels ]
        self.loss = tf.nn.seq2seq.sequence_loss(self.decode_outputs, self.labels, loss_weights, yvocab_size)

        # train op to minimize the loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        tf.logging.vlog(tf.logging.INFO, "Finished Initializing Model!")


    # get the feed dictionary
    def get_feed(self, X, Y, keep_prob):
        feed = {self.encoder_inputs[t]: X[t] for t in range(self.xseq_len)}
        feed.update({self.labels[t]: Y[t] for t in range(self.yseq_len)})
        # probability of a dropout
        feed[self.keep_prob] = keep_prob
        return feed

    # run one batch for training
    def train_batch(self, sess, train_batch_gen):
        # get batches
        batchX, batchY = train_batch_gen.__next__()
        # build feed
        feed = self.get_feed(batchX, batchY, keep_prob=0.5)
        _, loss_v = sess.run([self.train_op, self.loss], feed)
        return loss_v

    def eval_step(self, sess, eval_batch_gen):
        # get batches
        batchX, batchY = eval_batch_gen.__next__()
        # build feed
        feed = self.get_feed(batchX, batchY, keep_prob=1.)
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

    def train(self, train_set, valid_set, sess=None ):
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
                self.train_batch(sess, train_set)
                if i and i % (self.epochs // 100) == 0: 
                    # save model to disk
                    saver.save(sess, self.ckpt_path + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    val_loss = self.eval_batches(sess, valid_set, 16)
                    # print stats
                    print('Model saved to disk at iteration #{}'.format(i))
                    print('val loss : {0:.6f}'.format(val_loss))
                    sys.stdout.flush()
            except KeyboardInterrupt: # this will most definitely happen, so handle it
                tf.logging.vlog(tf.logging.INFO, "Interrupted by user at iteration {}".format(i))
                self.session = sess
                return sess

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    # prediction
    def predict(self, sess, X):
        assert(len(x) == self.xseq_len)
        feed_dict = {self.encoder_inputs[t]: X[t] for t in range(self.xseq_len)}
        feed_dict[self.keep_prob] = 1.
        dec_op_v = sess.run(self.decode_outputs_test, feed_dict)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)


