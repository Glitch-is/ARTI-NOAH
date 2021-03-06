import random
import nltk
import numpy as np
import tensorflow as tf
from noah.corpus.opensubsdata import OpensubsData
from noah.corpus.cornelldata import CornellData
from noah.corpus.ubuntudata import UbuntuData
import os
import pickle
import re
from tqdm import tqdm
from collections import Counter

class Dataset:
    def __init__(self, datasetPath, maxX=25, maxY=25, vocab_size=20000, corpus="txt", clean=True, partial=False):
        tf.logging.vlog(tf.logging.INFO, "Initializing Dataset...")
        self.datasetPath = datasetPath
        self.maxX = maxX
        self.maxY = maxY
        self.vocab_size = vocab_size
        self.corpus = corpus
        self.clean = clean
        self.partial = partial

        if corpus == "txt":
            self.savedSamplePath = "save/corpus/" + corpus + "-" + self.datasetPath.replace("/", "_") + "-" + str(maxX) + "-" + str(vocab_size) + ".pkl"
        else:
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
            self.tokens["GO"] = self.encodeWord("<GO>")
            assert(self.word2id["<go>"] == 0)
            self.tokens["PAD"] = self.encodeWord("<PAD>")
            self.tokens["END"] = self.encodeWord("<END>")
            self.tokens["UNKNOWN"] = self.encodeWord("<UNKNOWN>")

            if self.corpus == "txt":
                lines = []
                with open(self.datasetPath, "r") as f:
                    lines = f.read()

                lines = self.cleanText(lines).split("\n")

                for lineNum in tqdm(range(0, len(lines)-1, 2), total=len(lines)//2, desc="Processing..."):
                    self.process(lines[lineNum], lines[lineNum+1])

            else:
                if self.corpus == "opensubs":
                    osubs = OpensubsData(self.datasetPath)
                    conversations = osubs.getConversations()
                elif self.corpus == "cornell":
                    cornellData = CornellData(self.datasetPath)
                    conversations = cornellData.getConversations()
                elif self.corpus == "ubuntu":
                    ubuntuData = UbuntuData(self.datasetPath)
                    conversations = ubuntuData.getConversations()

                for conversation in tqdm(conversations, desc="Processing..."):
                    for i in range(len(conversation['lines']) - 1):
                        questionText = self.cleanText(conversation["lines"][i]["text"])
                        answerText = self.cleanText(conversation["lines"][i + 1]["text"])

                        if questionText != "" and answerText != "":
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
        padded_question = self.addPadding(question, self.maxX)[::-1]

        answer = self.extractText(answerText, answer=True)
        padded_answer = self.addPadding([self.tokens["GO"]] + answer + [self.tokens["END"]], self.maxY + 1)
        if question and answer:
            self.questions.append(padded_question)
            self.answers.append(padded_answer)

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
        for id in range(4, len(self.word2id)):
            count = idFreq[id]
            word = self.id2word[id]
            if count == 1:
                idMap[id] = self.tokens["UNKNOWN"]
                del self.word2id[word]
                del self.id2word[id]
            else:
                idMap[id] = stepCounter
                del self.id2word[id]
                self.word2id[word] = stepCounter
                self.id2word[stepCounter] = word
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
        if self.clean:
            return re.sub('[^a-zA-Z0-9 \n]','', text).strip()
        else:
            return text

    def addPadding(self, seq, l):
        return seq + [self.tokens["PAD"]] * (l - len(seq))


    def extractText(self, line, answer=False, store=True):

        sentence_tokens = nltk.sent_tokenize(line)
        if len(sentence_tokens) == 0:
            return []
        sentences = []
        for sent in sentence_tokens:
            words = nltk.word_tokenize(sent)
            sentences.append(words)

        mx = 0
        if answer:
            # -2 to account for the GO and EOS tokens
            mx = self.maxY - 2
        else:
            sentences.reverse()
            mx = self.maxX

        seq = []
        for sent in sentences:
            if len(seq) + len(sent) <= mx:
                if answer:
                    seq = seq + sent
                else:
                    seq = sent + seq
        if self.partial and not seq:
            if answer:
                seq = sentences[0][:mx]
            else:
                seq = sentences[0][len(sentences[0]) - mx:]

        seq = [self.encodeWord(word, store=store) for word in seq]

        return seq


    def encodeWord(self, word, store=True):
        word = word.lower()

        wordId = self.word2id.get(word)
        if not wordId:
            if not store:
                return self.tokens["UNKNOWN"]
            else:
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

    def separateY(self, y):
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
        assert(len(ydecoder[0]) == len(ylabels[0]) == self.maxY)
        return ydecoder, ylabels


    def getBatch(self, data, batch_size):
        q, a = data
        for i in range(0, len(data), batch_size):
            x, y = q[i:i + batch_size], a[i:i + batch_size]
            ydecoder, ylabels = self.separateY(y)
            yv = self.getYWeights(ylabels)
            yield x.T, ydecoder.T, ylabels.T, yv.T

    def getRandomBatch(self, data, batch_size):
        # a dooope trick
        q, a = data
        while True:
            s = random.sample(list(np.arange(len(q))), batch_size)
            # using a list to index a numpy matrix gives you the row vectors corresponding to that index
            x, y = q[s], a[s]
            ydecoder, ylabels = self.separateY(y)
            yv = self.getYWeights(ylabels)
            yield x.T, ydecoder.T, ylabels.T, yv.T

    def getBatches(self, data, batch_size):
        q, a = data
        def singleBatch():
            arr = np.arange(len(q))
            np.random.shuffle(arr)
            for i in range(0, len(arr), batch_size):
                s = arr[i:min(i + batch_size, len(arr))]
                x, y = q[s], a[s]
                assert(len(x[0]) == self.maxX)
                ydecoder, ylabels = self.separateY(y)
                yv = self.getYWeights(ylabels)
                assert(len(yv[0]) == self.maxY)
                # transpose because we want a[i] to be the vector for the word at i
                yield x.T, ydecoder.T, ylabels.T, yv.T
        class LenWrap(object):
            def __init__(self, gen):
                self.gen = gen
            def __iter__(self):
                return self
            def __next__(self):
                return self.gen.__next__()
            def __len__(self):
                return len(q) // batch_size
        while True:
            batch = singleBatch()
            yield LenWrap(batch)
