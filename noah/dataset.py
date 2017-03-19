import nltk
import tensorflow as tf

class Dataset:
    def __init__(self, datasetPath, maxLen=25, trainFrac=0.80):
        tf.logging.vlog(tf.logging.INFO, "Initializing DateSet...")
        self.datasetPath = datasetPath
        self.maxLen = maxLen
        self.trainFrac = trainFrac
        self.testFrac = 1 - trainFrac
        
        self.trainingExamples = []  # stored in the form of [[question,answer]]
        self.testingExamples = []

        self.tokens = {
            "GO": -1,
            "PAD": -1,
            "END": -1,
            "UNKNOWN": -1
        }

        self.word2id = {}
        self.id2word = {} # for fast reverse lookup

        self.loadData()
        tf.logging.vlog(tf.logging.INFO, "Loaded %d words and %d Q&A pairs" % (len(self.word2id), len(self.trainingExamples)))
        tf.logging.vlog(tf.logging.INFO, "Finished Initializing DateSet!")

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
        testSplit = int(len(lines) * self.trainFrac)
        # TODO: split into training and test data
        for lineNum in range(0, len(lines)-1, 2):
            question = self.extractText(lines[lineNum])
            answer = self.extractText(lines[lineNum+1])
            if lineNum <= testSplit:
                self.trainingExamples.append([question, answer])
            else:
                self.testingExamples.append([question, answer])

    def extractText(self, line):
        seq = []

        # TODO: iterate through sentences
        # TODO: handle if the word count is greater than the max size
        tokens = nltk.word_tokenize(line)
        for token in tokens:
            seq.append(self.encodeWord(token))
        seq.append(self.tokens["END"])

        # Add padding
        while len(seq) < self.maxLen:
            seq.append(self.tokens["PAD"])
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
