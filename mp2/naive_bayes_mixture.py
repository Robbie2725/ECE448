# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter,
                      bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    model = NaiveBayesMixModel(pos_prior, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter)
    model.train_phase(train_set, train_labels)
    return model.dev_phase(dev_set)


class NaiveBayesMixModel:
    def __init__(self, pos_prior, bigram_lambda, unigram_smoothing, bigram_smoothing):
        self.b_lambda = bigram_lambda
        self.u_smoothing = unigram_smoothing
        self.b_smoothing = bigram_smoothing
        self.p_type = {
            'spam': 1-pos_prior,
            'ham': pos_prior
        }
        self.p_word_b = {
            'ham': {},
            'spam': {}
        }
        self.p_word_u = {
            'ham': {},
            'spam': {}
        }
        self.word_ct = {
            'ham': 0,
            'spam': 0
        }
        self.bigram_ct = {
            'ham': 0,
            'spam': 0
        }
        self.unique_b = {}
        self.unique_u = {}

    def train_phase(self, train_set, train_labels):
        word_ct_ham = 0
        word_ct_spam = 0
        bigram_ct_ham = 0
        bigram_ct_spam = 0

        # loop through all training sets
        for i in range(len(train_set)):
            label = train_labels[i]  # get the set label
            set_ = train_set[i]  # get the set

            for j in range(len(set_)):
                word = set_[j]  # get the current word
                self.unique_u[word] = 1  # track number of unique words

                # if not the last word in the set, then get the bigram
                if j == len(set_) - 1:
                    bigram = None
                else:
                    bigram = (set_[j], set_[j + 1])

                if label:
                    # this is a ham email
                    word_ct_ham += 1  # increase the word count for ham

                    # increase this current words count
                    if word in self.p_word_u['ham']:
                        self.p_word_u['ham'][word] += 1
                    else:
                        self.p_word_u['ham'][word] = 1

                    # if there is a bigram
                    if bigram is not None:
                        bigram_ct_ham += 1  # increase the bigram count for ham emails
                        self.unique_b[bigram] = 1  # track number of unique bigrams
                        # increase the count of this bigram
                        if bigram in self.p_word_b['ham']:
                            self.p_word_b['ham'][bigram] += 1
                        else:
                            self.p_word_b['ham'][bigram] = 1
                else:
                    # This is a spam email
                    word_ct_spam += 1  # increase the total spam word count

                    # increase this current owrds count
                    if word in self.p_word_u['spam']:
                        self.p_word_u['spam'][word] += 1
                    else:
                        self.p_word_u['spam'][word] = 1

                    # if there is a bigram
                    if bigram is not None:
                        self.unique_b[bigram] = 1  # track number of unique bigrams
                        bigram_ct_spam += 1  # increase the bigram count for spam emails

                        # increase the count of this bigram
                        if bigram in self.p_word_b['spam']:
                            self.p_word_b['spam'][bigram] += 1
                        else:
                            self.p_word_b['spam'][bigram] = 1

        # store the bigram/unigram counts
        self.bigram_ct['ham'] += bigram_ct_ham
        self.bigram_ct['spam'] += bigram_ct_spam
        self.word_ct['ham'] += word_ct_ham
        self.word_ct['spam'] += word_ct_spam

    def dev_phase(self, dev_set):
        label_list = []  # the list to return

        # loop through all emails in the development set
        for set_ in dev_set:

            # initialize variables for unigram/bigram probabilities
            p_word_spam = 0
            p_word_ham = 0
            p_bigram_spam = 0
            p_bigram_ham = 0
            for i in range(len(set_)):
                word = set_[i]
                # add to the ML estimate for spam/ham unigrams with laplace smoothing
                p_word_spam += math.log((self.p_word_u['spam'].get(word, 0) + self.u_smoothing)) - math.log(
                    self.word_ct['spam'] + self.u_smoothing * len(self.unique_u.keys()))
                p_word_ham += math.log((self.p_word_u['ham'].get(word, 0) + self.u_smoothing)) - math.log(
                    self.word_ct['ham'] + self.u_smoothing * len(self.unique_u.keys()))

                # check if there is a bigram left
                if i == len(set_) - 1:
                    continue

                bigram = (set_[i], set_[i + 1])  # get the bigram
                # add to the ML estimate for spam/ham bigrams with laplace smoothing
                p_bigram_spam += math.log((self.p_word_b['spam'].get(bigram, 0) + self.b_smoothing)) - math.log(
                        self.bigram_ct['spam'] + self.b_smoothing * len(self.unique_b.keys()))
                p_bigram_ham += math.log((self.p_word_b['ham'].get(bigram, 0) + self.b_smoothing)) - math.log((
                        self.bigram_ct['ham'] + self.b_smoothing * len(self.unique_b.keys())))

            # Calculate P(ham| words) and P(spam|words) for this dev set
            p_ham_words = (1 - self.b_lambda) * (math.log(self.p_type['ham']) + p_word_ham) + self.b_lambda * (math.log(
                self.p_type['ham']) + p_bigram_ham)
            p_spam_words = (1 - self.b_lambda) * (math.log(self.p_type['spam']) + p_word_spam) + self.b_lambda * (
                        math.log(self.p_type['spam']) + p_bigram_spam)

            # mark the label depending on P(ham| words) and P(spam|words)
            if p_ham_words >= p_spam_words:
                label_list.append(1)  # if P(ham|words) > P(spam|words), mark as ham
            else:
                label_list.append(0)  # else mark as spam (i.e. P(ham|words) < P(spam|words))

        return label_list
