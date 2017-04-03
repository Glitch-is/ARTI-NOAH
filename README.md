# NOAH: Neural Oriented Artificial Human
The goal of this project was to replicate the results of "A Neural Conversational Model" by Google, where a neural network is used to provide responses to inputs. We attempt to implement the model and reproduce some of the results, in addition to training the network on our own data.

# Goals
The goals of this project were to answer the following questions:
* What does a neural network predict on our data?
* Is it possible to train a neural network to answer contextually?
* What effect does the quality of datasets have on the quality of responses?
The project also had a lot of learning opportunities we had not been able to review before. These included learning a basic understanding of neural networks, using the TensorFlow library,data processing, utilizing NLTK to process natural language, setting up training instances, and etc.

# Implementation
This project was implemented using Google’s TensorFlow library for deep learning, along with NLTK for processing our datasets and Numpy for additional mathematics.The models were trained on server using an Nvidia Tesla K80 GPU.

# Results
Here's a chat bot we trained for 30 minutes on Cornell's Movie-Dialogue Corpus.
```
Human: hi
Machine: hi , david .
Human: what 's up ?
Machine: nothing .
Human: is this sentence false ?
Machine: no .
Human: can you say no ?
Machine: sure .
Human: can you say yes ?
Machine: i 'm sorry .
Human: can you say hello ?
Machine: huh ?
Human: how old are you ?
Machine: thirty-five .
Human: when are you ready ?
Machine: tomorrow .
Human: are you alive ?
Machine: no .
Human: what is your name ?
Machine: peggy sue <unknown> .
Human: say `` table '' .
Machine: fuck you .
Human: how much ?
Machine: $ 20 .
Human: how many ?
Machine: three .
Human: what color ?
Machine: gray .
Human: what time ?
Machine: six o'clock .
Human: what year ?
Machine: nineteen fifty
```

# Authors
* Hlynur Óskar Guðmundsson
* James Elías Sigurðarson
* Arnar Páll Jóhannsson

# References
[A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
[DeepQA](https://github.com/Conchylicultor/DeepQA)
