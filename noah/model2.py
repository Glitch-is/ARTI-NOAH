# Copyright 2015 Conchylicultor. All Rights Reserved.
# Modifications copyright (C) 2016 Carlos Segura
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Model to predict the next sentence given an input sequence

"""

import tensorflow as tf
import numpy as np
import sys
import os
import math




class Model:
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, **kwargs):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        kwargs['dropout_prob'] = kwargs.get('dropout_prob', 0.9)
        print("Model creation...")
        class objview(object):
            def __init__(self, d):
                self.__dict__ = d
        self.args = objview(kwargs)

        self.dtype = tf.float32

        # Placeholders
        self.encoderInputs  = None
        self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        self.decoderTargets = None
        self.decoderWeights = None  # Adjust the learning to the target sentence size

        # Main operators
        self.lossFct = None
        self.optOp = None
        self.outputs = None  # Outputs of the network, list of probability for each words

        os.makedirs(self.args.save_path, exist_ok=True)

        # Construct the graphs
        self.buildNetwork()

    def buildNetwork(self):
        """ Create the computational graph
        """

        # TODO: Create name_scopes (for better graph visualisation)
        # TODO: Use buckets (better perfs)

        # Creation of the rnn cell
        def create_rnn_cell():
            encoDecoCell = tf.contrib.rnn.BasicLSTMCell(  # Or GRUCell, LSTMCell(args.hiddenSize)
                self.args.hidden_size,
            )
            if self.args.train:  # TODO: Should use a placeholder instead
                encoDecoCell = tf.contrib.rnn.DropoutWrapper(
                    encoDecoCell,
                    input_keep_prob=1.0,
                    output_keep_prob=self.args.dropout_prob
                )
            return encoDecoCell
        encoDecoCell = tf.contrib.rnn.MultiRNNCell(
            [create_rnn_cell() for _ in range(self.args.num_layers)],
        )

        # Network input (placeholders)

        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.xlen)]  # Batch size * sequence length * input dim

        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.ylen)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.ylen)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.ylen)]

        # Define the network
        # Here we use an embedding model, it takes integer as input and convert them into word vector for
        # better word representation
        decoderOutputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,  # List<[batch=?, inputDim=1]>, list of size args.maxLength
            self.decoderInputs,  # For training, we force the correct output (feed_previous=False)
            encoDecoCell,
            self.args.xsize,
            self.args.ysize,  # Both encoder and decoder have the same number of class
            embedding_size=self.args.embedding_size,  # Dimension of each word
            output_projection=None,
            feed_previous=not bool(self.args.train)  # When we test (self.args.test), we use previous output as next input (feed_previous)
        )

        # TODO: When the LSTM hidden size is too big, we should project the LSTM output into a smaller space (4086 => 2046): Should speed up
        # training and reduce memory usage. Other solution, use sampling softmax

        # For testing only
        if not self.args.train:
            self.outputs = decoderOutputs

        # For training only
        else:
            # Finally, we define the loss function
            self.lossFct = tf.contrib.legacy_seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.args.ysize,
                softmax_loss_function=None  # If None, use default SoftMax
            )
            tf.summary.scalar('loss', self.lossFct)  # Keep track of the cost

            # Initialize the optimizer
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learning_rate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)

    def step(self, batch):
        """ Forward/training step operation.
        Does not perform run on itself but just return the operators to do so. Those have then to be run
        Args:
            batch (Batch): Input data on testing mode, input and target on output mode
        Return:
            (ops), dict: A tuple of the (training, loss) operators or (outputs,) in testing mode with the associated feed dictionary
        """

        # Feed the dictionary
        inputX, inputY, inputY2, weightsY = batch
        feedDict = {}
        ops = None

        if self.args.train:  # Training
            for i in range(self.args.xlen):
                feedDict[self.encoderInputs[i]]  = inputX[i]
            for i in range(self.args.ylen):
                feedDict[self.decoderInputs[i]]  = inputY2[i]
                feedDict[self.decoderTargets[i]] = inputY[i]
                feedDict[self.decoderWeights[i]] = weightsY[i]

            ops = (self.optOp, self.lossFct)
        else:  # Testing (batchSize == 1)
            for i in range(self.args.xlen):
                feedDict[self.encoderInputs[i]]  = inputX[i]
            feedDict[self.decoderInputs[0]]  = [0] * len(inputX[0])

            ops = (self.outputs,)

        # Return one pass operator
        return ops, feedDict
    def train(self, train_set, valid_set, sess=None, save_every=50):
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
        step = 0
        for i in range(self.args.epochs):
            print("Epoch {}/{}".format(step+1, self.args.epochs))
            try:
                batch = train_set.__next__()
                for X, Y, Yv in batch:
                    Y2 = []
                    for y in Y.T:
                        Y2.append(np.insert(y, 0, 0)[:-1])
                    Y2 = np.array(Y2).T
                    ops, feed = self.step((X, Y, Y2, Yv))
                    _, loss = sess.run(ops, feed)
                    step += 1
                    if step % 100 == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                        print("Step %d Loss %.2f Perplexity %.2f" % (step, loss, perplexity))
                    if step % save_every == 0:
                        # save model to disk
                        saver.save(sess, self.args.save_path + self.args.model_name + '.ckpt', global_step=i)
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

        saver.save(sess, self.args.save_path + self.args.model_name + '.ckpt', global_step=i)
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
        ckpt = tf.train.get_checkpoint_state(self.args.save_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess
    # prediction
    def predict(self, sess, X):
        assert(not self.args.train)
        assert(len(X) == self.args.xlen)
        ops, feed = self.step((X, None, None, None))

        dec_op_v = sess.run(ops[0], feed)
        print(dec_op_v)
        # dec_op_v is a list; also need to transpose 0,1 indices 
        #  (interchange batch_size and timesteps dimensions)
        dec_op_v = np.array(dec_op_v).transpose([1,0,2])
        # return the index of item with highest probability
        return np.argmax(dec_op_v, axis=2)
