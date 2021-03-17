# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

import math

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    model = NaiveBayesModel(pos_prior, smoothing_parameter)
    model.train_phase(train_set, train_labels)
    return model.dev_phase(dev_set)


class NaiveBayesModel:
    def __init__(self, prior, smoothing):
        self.p_word = {}
        self.p_type = {
            'spam': 1-prior,
            'ham': prior
        }
        self.word_ct_spam = 0
        self.word_ct_ham = 0
        self.smoothing = smoothing

    def train_phase(self, train_set, train_labels):
        # used to count number of words in each class
        word_ct_ham = 0
        word_ct_spam = 0
        # current index for train_labels
        idx = 0

        # loop through each email
        for set_ in train_set:

            # get the label of this email
            label = train_labels[idx]
            idx += 1

            # count occurences of each word in this set by looping through each word in the current email
            for word in set_:
                if label:
                    # This is a ham email
                    word_ct_ham += 1  # increase total ham word count

                    # increase count of the word for ham
                    if word in self.p_word:
                        self.p_word[word]['ham'] += 1
                    else:
                        self.p_word[word] = {'ham': 1, 'spam': 0}
                else:
                    # This is a spam email
                    word_ct_spam += 1  # increase total spam word count

                    # increase count of the word
                    if word in self.p_word:
                        self.p_word[word]['spam'] += 1
                    else:
                        self.p_word[word] = {'ham': 0, 'spam': 1}

        # save the total ham/spam word counts
        self.word_ct_ham += word_ct_ham
        self.word_ct_spam += word_ct_spam

    def dev_phase(self, dev_set):
        label_list = []  # the list to return

        # loop through all emails in development set
        for set_ in dev_set:
            p_word_spam = 0
            p_word_ham = 0
            # calculate the ML estimate for spam and ham with laplace smoothing
            for word in set_:
                p_word_spam += math.log((self.p_word.get(word, {'spam': 0}).get('spam') + self.smoothing)) - math.log(
                            self.word_ct_spam + self.smoothing * (len(self.p_word.keys())))
                p_word_ham += math.log((self.p_word.get(word,  {'ham': 0}).get('ham') + self.smoothing)) - math.log(
                            self.word_ct_ham + self.smoothing * (len(self.p_word.keys())))

            # calculate P(ham| words) and P(spam|words)
            p_ham_word = math.log(self.p_type['ham']) + p_word_ham
            p_spam_word = math.log(self.p_type['spam']) + p_word_spam
            if p_ham_word >= p_spam_word:
                label_list.append(1)  # if P(ham|words) > P(spam|words), mark as ham
            else:
                label_list.append(0)  # else mark as spam (i.e. P(ham|words) < P(spam|words))
        return label_list
