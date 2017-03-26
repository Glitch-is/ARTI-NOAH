import random
import nltk
import numpy as np
import tensorflow as tf

class Dataset:
    def __init__(self, datasetPath, maxX=25, maxY=25, trainFrac=0.80):
        tf.logging.vlog(tf.logging.INFO, "Initializing DateSet...")
        self.datasetPath = datasetPath
        self.maxX = maxX
        self.maxY = maxY
        self.trainFrac = trainFrac
        self.testFrac = 1 - trainFrac

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
        tf.logging.vlog(tf.logging.INFO, "Finished Initializing DateSet!")

    def getTrainingData(self):
        testSplit = int(len(self.questions) * self.trainFrac)
        return self.questions[:testSplit], self.answers[:testSplit]

    def getTestingData(self):
        testSplit = int(len(self.questions) * self.trainFrac)
        return self.questions[testSplit:], self.answers[testSplit:]

    def loadData(self):
        self.tokens["GO"] = self.encodeWord("<GO>")
        assert(self.word2id["<go>"] == 0)
        self.tokens["PAD"] = self.encodeWord("<PAD>")
        self.tokens["END"] = self.encodeWord("<END>")
        self.tokens["UNKNOWN"] = self.encodeWord("<UNKNOWN>")

        lines = []
        #TODO: take dataset file as parameter
        with open(self.datasetPath, "r") as f:
            lines = f.read().split("\n")
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

    def addPadding(self, seq):
        return seq + [self.tokens["PAD"]] * (self.maxX - len(seq))


    def extractText(self, line, answer=False):
        seq = []

        # TODO: iterate through sentences
        # TODO: handle if the word count is greater than the max size
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
            seq.append(self.encodeWord(token))

        return seq


    def encodeWord(self, word):
        word = word.lower()

        wordId = self.word2id.get(word)
        if not wordId:
            wordId = len(self.word2id)

        self.word2id[word] = wordId
        self.id2word[wordId] = word

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

    def getBatch(self, data, batch_size):
        q, a = data
        for i in range(0, len(w), batch_size):
            x, y = q[i:i + batch_size], a[i:i + batch_size]
            yield x.T, y.T

    def getRandomBatch(self, data, batch_size):
        # a dooope trick
        q, a = data
        while True:
            print(len(q), batch_size)
            s = random.sample(list(np.arange(len(q))), batch_size)
            # using a list to index a numpy matrix gives you the row vectors corresponding to that index
            x, y = q[s], a[s]
            # transpose because we want a[i] to be the vector for the word at i
            yield x.T, y.T


