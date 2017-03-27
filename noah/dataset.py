import random
import nltk
import numpy as np
import tensorflow as tf
from noah.corpus.opensubsdata import OpensubsData
import os
import pickle

class Dataset:
    def __init__(self, datasetPath, maxX=25, maxY=25, trainFrac=0.80, vocab_size=20000, corpus="txt"):
        tf.logging.vlog(tf.logging.INFO, "Initializing Dataset...")
        self.datasetPath = datasetPath
        self.maxX = maxX
        self.maxY = maxY
        self.trainFrac = trainFrac
        self.testFrac = 1 - trainFrac
        self.vocab_size = vocab_size
        self.corpus = corpus

        self.savedSamplePath = "save/corpus/" + corpus + "-" + str(maxX) + "-" + str(vocab_size) + ".pkl"

        self.questions = []
        self.answers = []

        self.tokens = {
            "GO": -1,
            "PAD": -1,
            "END": -1,
            "UNKNOWN": -1
        }

        self.word2id = {}
        self.id2word = {} # for fast reverse lookup

        self.loadData()
        tf.logging.vlog(tf.logging.INFO, "Loaded %d words and %d Q&A pairs" % (len(self.word2id), len(self.questions)))
        tf.logging.vlog(tf.logging.INFO, "Finished Initializing Dataset!")

    def getTrainingData(self):
        testSplit = int(len(self.questions) * self.trainFrac)
        return self.questions[:testSplit], self.answers[:testSplit]

    def getTestingData(self):
        testSplit = int(len(self.questions) * self.trainFrac)
        return self.questions[testSplit:], self.answers[testSplit:]

    def loadData(self):
        self.tokens["GO"] = self.encodeWordStore("<GO>")
        assert(self.word2id["<go>"] == 0)
        self.tokens["PAD"] = self.encodeWordStore("<PAD>")
        self.tokens["END"] = self.encodeWordStore("<END>")
        self.tokens["UNKNOWN"] = self.encodeWordStore("<UNKNOWN>")

        if os.path.isfile(self.savedSamplePath):
            with open(self.savedSamplePath, 'rb') as f:
                data = pickle.load(f)
                self.word2id = data['word2id']
                self.id2word = data['id2word']
                self.questions = data['questions']
                self.answers = data['answers']

            self.tokens["PAD"] = self.word2id['<pad>']
            self.tokens["GO"] = self.word2id['<go>']
            self.tokens["END"] = self.word2id['<end>']
            self.tokens["UNKNOWN"] = self.word2id['<unknown>']
        else:
            if self.corpus == "txt":
                lines = []
                with open(self.datasetPath, "r") as f:
                    lines = f.read()
                # dist = nltk.FreqDist(nltk.word_tokenize(lines.lower()))
                # print("Total word count:")
                # print(len(dist))
                #
                # vocab = [x[0] for x in dist.most_common(self.vocab_size)]
                # print(vocab)

                # print("Vocab: ")
                # print(len(vocab))
                #
                # for word in vocab:
                #     self.encodeWordStore(word)
                #
                # print("Word2id: ")
                # print(len(self.word2id))

                lines = lines.replace(".", "").replace(",", "").replace("'", "").replace("?", "").replace("!", "").split("\n")

                # TODO: split into training and test data
                for lineNum in range(0, len(lines)-1, 2):
                    question = self.extractText(lines[lineNum])
                    question = self.addPadding(question)[::-1]
                    self.questions.append(question)

                    answer = self.extractText(lines[lineNum+1], answer=True)
                    answer = self.addPadding(answer + [self.tokens["END"]])
                    self.answers.append(answer)

                self.questions = np.array(self.questions)
                self.answers = np.array(self.answers)

            elif self.corpus == "opensubs":
                osubs = OpensubsData(self.datasetPath)
                conversations = osubs.getConversations()
                for conversation in conversations:
                    question = self.extractText(conversation["lines"][0]["text"])
                    question = self.addPadding(question)[::-1]
                    self.questions.append(question)

                    answer = self.extractText(conversation["lines"][1]["text"], answer=True)
                    answer = self.addPadding(answer + [self.tokens["END"]])
                    self.answers.append(answer)

            self.questions = np.array(self.questions)
            self.answers = np.array(self.answers)

            with open(os.path.join(self.savedSamplePath), 'wb') as f:
                data = {
                    'word2id': self.word2id,
                    'id2word': self.id2word,
                    'questions': self.questions,
                    'answers': self.answers
                }
                pickle.dump(data, f, -1)

    def addPadding(self, seq):
        return seq + [self.tokens["PAD"]] * (self.maxX - len(seq))


    def extractText(self, line, answer=False):
        seq = []

        sentence_tokens = nltk.sent_tokenize(line)
        if answer:
            line = sentence_tokens[0]
        else:
            line = sentence_tokens[-1]

        tokens = nltk.word_tokenize(line)
        if answer: 
            if len(tokens) >= (self.maxY - 1):
                tokens = tokens[:self.maxY - 1]
        else:
            if len(tokens) >= self.maxX:
                tokens = tokens[:self.maxX]

        for token in tokens:
            seq.append(self.encodeWordStore(token))

        return seq


    def encodeWordStore(self, word):
        word = word.lower()

        wordId = self.word2id.get(word)
        if not wordId:
            wordId = len(self.word2id)

        self.word2id[word] = wordId
        self.id2word[wordId] = word

        return wordId

    def encodeWord(self, word):
        word = word.lower()

        wordId = self.word2id.get(word)
        if not wordId:
            wordId = self.tokens["UNKNOWN"]

        return wordId

    def decodeId(self, idNum):
        return self.id2word[idNum]

    def sequence2str(self, seq):
        ret = []
        for id in seq:
            if id == self.tokens["END"]:
                break
            if id != self.tokens["PAD"] and id != self.tokens["GO"]:
                ret.append(self.id2word[id])
        return " ".join(ret)

    def getYWeights(self, Y):
        yweights = [np.ones(len(a), dtype=np.float32) for a in Y]
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] == self.tokens["PAD"]:
                    yweights[i][j] = 0.0
        return np.array(yweights)


    def getBatch(self, data, batch_size):
        q, a = data
        for i in range(0, len(data), batch_size):
            x, y = q[i:i + batch_size], a[i:i + batch_size]
            yv = self.getYWeights(y)
            yield x.T, y.T, yv.T

    def getRandomBatch(self, data, batch_size):
        # a dooope trick
        q, a = data
        while True:
            s = random.sample(list(np.arange(len(q))), batch_size)
            # using a list to index a numpy matrix gives you the row vectors corresponding to that index
            x, y = q[s], a[s]
            yv = self.getYWeights(y)
            # transpose because we want a[i] to be the vector for the word at i
            yield x.T, y.T, yv.T

    def getBatches(self, data, batch_size):
        batches = []
        for batch in self.getBatch(data, batch_size):
            batches.append(batch)

        return batches
