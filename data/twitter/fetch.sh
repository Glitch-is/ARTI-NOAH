#!/bin/bash
wget https://raw.githubusercontent.com/Marsan-Ma/chat_corpus/master/twitter_en.txt.gz
gunzip twitter_en.txt.gz
head -n 50000 twitter_en.txt > twitter_en_small.txt
