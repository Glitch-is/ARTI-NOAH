import random
import nltk
import numpy as np
import tensorflow as tf
from noah.corpus.opensubsdata import OpensubsData
from noah.corpus.cornelldata import CornellData
import os
import pickle
import re
from tqdm import tqdm
from collections import Counter

class Dataset:
    def __init__(self, datasetPath, maxX=25, maxY=25, vocab_size=20000, corpus="txt"):
        tf.logging.vlog(tf.logging.INFO, "Initializing Dataset...")
        self.datasetPath = datasetPath
        self.maxX = maxX
        self.maxY = maxY
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
        return self.questions, self.answers

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

                lines = self.cleanText(lines)

                for lineNum in tqdm(range(0, len(lines)-1, 2), total=len(lines)//2, desc="Processing..."):
                    self.process(lines[lineNum], lines[lineNum+1])

            else:
                if self.corpus == "opensubs":
                    osubs = OpensubsData(self.datasetPath)
                    conversations = osubs.getConversations()
                elif self.corpus == "cornell":
                    cornellData = CornellData(self.datasetPath)
                    conversations = cornellData.getConversations()
                for conversation in tqdm(conversations, desc="Processing..."):
                    questionText = self.cleanText(conversation["lines"][0]["text"])
                    answerText = self.cleanText(conversation["lines"][1]["text"])

                    if questionText != "" and answerText != "":
                        if not questionText.isspace() and not answerText.isspace():
                            self.process(questionText, answerText)

            self.questions = np.array(self.questions)
            self.answers = np.array(self.answers)

            # Get rid of words with only one occurance and fix ids
            self.pruneData()

            os.makedirs(os.path.dirname(self.savedSamplePath), exist_ok=True)
            with open(os.path.join(self.savedSamplePath), 'wb') as f:
                data = {
                    'word2id': self.word2id,
                    'id2word': self.id2word,
                    'questions': self.questions,
                    'answers': self.answers
                }
                pickle.dump(data, f, -1)


    def process(self, questionText, answerText):
        question = self.extractText(questionText)
        question = self.addPadding(question, self.maxX)[::-1]
        self.questions.append(question)

        answer = self.extractText(answerText, answer=True)
        answer = self.addPadding([self.tokens["GO"]] + answer + [self.tokens["END"]], self.maxY + 1)
        self.answers.append(answer)

    def pruneData(self):
        idFreq = Counter()
        tokens = set(self.tokens.values())
        for sequence in self.answers:
            for wordId in sequence:
                if wordId in tokens:
                    continue
                idFreq[wordId] += 1
        for sequence in self.questions:
            for wordId in sequence:
                if wordId in tokens:
                    continue
                idFreq[wordId] += 1

        idMap = {}
        # assume the tokens are at the front
        stepCounter = len(tokens) 
        for (id, count) in idFreq.items():
            word = self.id2word[id]
            if count == 1:
                idMap[id] = self.tokens["UNKNOWN"]
                del self.word2id[word]
                del self.id2word[id]
            else:
                idMap[id] = stepCounter
                self.word2id[word] = stepCounter
                self.id2word[stepCounter] = word
                del self.id2word[id]
                stepCounter += 1

        for sequence in self.answers:
            for index, wordId in enumerate(sequence):
                # ignore tokens
                if wordId in tokens:
                    continue
                sequence[index] = idMap[wordId]
        for sequence in self.questions:
            for index, wordId in enumerate(sequence):
                # ignore tokens
                if wordId in tokens:
                    continue
                sequence[index] = idMap[wordId]

    def cleanText(self, text):
        return re.sub('[^a-zA-Z0-9 ]','', text)

    def addPadding(self, seq, l):
        return seq + [self.tokens["PAD"]] * (l - len(seq))


    def extractText(self, line, answer=False, store=True):
        seq = []

        sentence_tokens = nltk.sent_tokenize(line)
        if len(sentence_tokens) == 0:
            return []
        if answer:
            line = sentence_tokens[0]
        else:
            line = sentence_tokens[-1]

        tokens = nltk.word_tokenize(line)
        if answer:
            # -2 to account for the GO and EOS tokens
            if len(tokens) >= (self.maxY - 2):
                tokens = tokens[:self.maxY - 2]
        else:
            if len(tokens) >= self.maxX:
                tokens = tokens[:self.maxX]

        for token in tokens:
            if store:
                seq.append(self.encodeWordStore(token))
            else:
                seq.append(self.encodeWord(token))

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

    def str2sequence(self, string):
        seq = self.extractText(self.cleanText(string), store=False)
        res = self.addPadding(seq, self.maxX)[::-1]
        return np.array([res])

    def getYWeights(self, Y):
        yweights = [[] for a in Y]
        for i in range(len(Y)):
            for j in range(self.maxY):
                if Y[i][j] == self.tokens["PAD"]:
                    break
            yweights[i] = [1.0] * j + [0.0] * (self.maxY - j)
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
        q, a = data
        def singleBatch():
            arr = np.arange(len(q))
            np.random.shuffle(arr)
            for i in range(0, len(arr), batch_size):
                s = arr[i:min(i + batch_size, len(arr))]
                x, y = q[s], a[s]
                # There is an extra padding symbol in there
                assert(len(y[0]) == self.maxY + 1)
                # ydecoder is the input without the extra padding
                # ylabels is the input without the go symbol
                ydecoder = []
                ylabels = []
                for i in range(len(y)):
                    ydecoder.append(y[i][:-1])
                    ylabels.append(y[i][1:])
                ydecoder = np.array(ydecoder)
                ylabels = np.array(ylabels)
                assert(len(x[0]) == self.maxX)
                assert(len(ydecoder[0]) == len(ylabels[0]) == self.maxY)
                yv = self.getYWeights(ylabels)
                assert(len(yv[0]) == self.maxY)
                # transpose because we want a[i] to be the vector for the word at i
                yield x.T, ydecoder.T, ylabels.T, yv.T
        while True:
            yield tqdm(singleBatch(), total=len(q) // batch_size)
