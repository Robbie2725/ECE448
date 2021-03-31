# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

import operator

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    pred = {}
    total_label_ct = {}
    for sentence in train:
        for pair in sentence:
            cur_count_dict = pred.setdefault(pair[0], {})
            label_ct = cur_count_dict.get(pair[1], 0)
            cur_count_dict[pair[1]] = label_ct+1
            pred[pair[0]] = cur_count_dict

            label_ct_total = total_label_ct.setdefault(pair[1], 0)
            total_label_ct[pair[1]] = label_ct_total+1

    test_labels = []
    # Found this utility code for getting the key with max value in a dictionary at:
    # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    most_often_tag = max(total_label_ct.items(), key=operator.itemgetter(1))[0]

    for sentence in test:
        test_labels.append([])
        sentence_idx = len(test_labels)-1
        for word in sentence:
            if word in pred.keys():
                tag_ct = pred[word]
                test_labels[sentence_idx].append((word, max(tag_ct.items(), key=operator.itemgetter(1))[0]))
            else:
                test_labels[sentence_idx].append((word, most_often_tag))
    return test_labels