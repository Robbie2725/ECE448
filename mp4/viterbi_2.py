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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math
import operator

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_ct = {}  # counts total occurances of each tag
    tag_pair_ct = {}  # for each tag, holds a dict counting the # of occurences it followed each tag
    tag_word_ct = {}  # counts the number of occurances of a word with each tag
    hapax_count = {}

    # Do the counts
    for sentence in train:
        for i in range(len(sentence)):
            tw_pair = sentence[i]

            # count occurences of tag/word pairs
            cur_word = tw_pair[0]
            cur_tag = tw_pair[1]
            cur_tw_dict = tag_word_ct.setdefault(cur_word, {})
            cur_tw_ct = cur_tw_dict.setdefault(cur_tag, 0)
            cur_tw_dict[cur_tag] = cur_tw_ct + 1
            tag_word_ct[cur_word] = cur_tw_dict

            # Count the number of times this word has appeared
            word_ct = hapax_count.get(cur_word, 0)
            hapax_count[cur_word] = word_ct + 1

            # count occurances of tags
            cur_tag_ct = tag_ct.setdefault(cur_tag, 0)
            tag_ct[cur_tag] = cur_tag_ct + 1

            # count occurances of tag pairs
            if i == len(sentence) - 1:
                continue
            next_tw = sentence[i + 1]
            next_tag = next_tw[1]
            next_tag_p = tag_pair_ct.setdefault(next_tag, {})
            cur_tag_p_ct = next_tag_p.setdefault(cur_tag, 0)
            next_tag_p[cur_tag] = cur_tag_p_ct + 1
            tag_pair_ct[next_tag] = next_tag_p

    k = 1
    hapax_tags = {}
    hapax_words = []
    for word in hapax_count.keys():
        if hapax_count[word] == 1:
            hapax_words.append(word)
            hapax_tag = list(tag_word_ct[word].keys())[0]
            # hapax_tags[hapax_tag] = 1
            h_tag_ct = hapax_tags.get(hapax_tag, 0)
            hapax_tags[hapax_tag] = h_tag_ct + 1

    p_t_hap = {}
    for tag in tag_ct.keys():
        p_t_hap[tag] = (hapax_tags.get(tag, 0)+k)/(len(hapax_words)+k*(len(tag_ct.keys())+1))

    k = 10**-5
    # compute the smoothed probabilities
    p_tb_ta = {}
    # compute probability t_b followed t_a (i.e. P(t_b|t_a))
    unique_tb_ct = len(tag_ct.keys())
    for tb in tag_ct.keys():
        for ta in tag_ct.keys():
            tb_ta_ct = tag_pair_ct.get(tb, {}).get(ta, 0)
            ta_ct = tag_ct[ta]
            p_tb_ta[(tb, ta)] = math.log(tb_ta_ct + k) - math.log(ta_ct + k * (unique_tb_ct + 1))

    # compute probablility tag yield word (i.e. P(W|t) )
    p_w_t = {}
    unique_w_ct = len(tag_word_ct.keys())
    for word in tag_word_ct.keys():
        for t in tag_ct.keys():
            w_t_ct = tag_word_ct.get(word, {}).get(t, 0)
            t_ct = tag_ct[t]
            if hapax_count[word] == 1:
                k_scale = k * p_t_hap[t]
            else:
                k_scale = k
            p_w_t[(word, t)] = math.log(w_t_ct + k_scale * p_t_hap[t]) - math.log(t_ct + k_scale * p_t_hap[t] * (unique_w_ct + 1))

    tags = tag_ct.keys()
    log_p_zero = -10000000  # use this for P = 0 since log(0) is undefined
    test_labels = []
    for sentence in test:
        # construct the trellis as a list of dict
        trellis_states = []
        trellis_bptr = []
        # for t=1
        trellis_states.append({})
        trellis_bptr.append({})
        for tag in tags:
            trellis_bptr[0][tag] = None
            if tag == 'START':
                trellis_states[0][tag] = math.log(1)
            else:
                trellis_states[0][tag] = log_p_zero

        # for t>1
        for i in range(1, len(sentence)):
            trellis_states.append({})
            trellis_bptr.append({})
            cur_word = sentence[i]

            # compute probabilities for each tag at time = i
            for new_tag in tags:

                # Find the tags probability, store the backpointer
                max_p = None
                max_p_prev_tag = None
                for prev_tag in tags:
                    if (cur_word, new_tag) not in p_w_t.keys():
                        scale_k = k * p_t_hap[new_tag]
                        p_prev_word_tag = math.log(scale_k) - math.log(tag_ct[new_tag] + scale_k * (unique_w_ct + 1))
                    else:
                        p_prev_word_tag = p_w_t[(cur_word, new_tag)]
                    cur_p = trellis_states[i - 1][prev_tag] + p_tb_ta[(new_tag, prev_tag)] + p_prev_word_tag
                    if max_p is None:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                    elif cur_p > max_p:
                        max_p = cur_p
                        max_p_prev_tag = prev_tag
                trellis_states[i][new_tag] = max_p
                trellis_bptr[i][new_tag] = max_p_prev_tag
        # Now backtrack
        sentence_list = []
        state_idx = len(trellis_bptr) - 1
        highest_p_state = max(trellis_states[state_idx].items(), key=operator.itemgetter(1))[0]
        while highest_p_state is not None:
            sentence_list.append((sentence[state_idx], highest_p_state))
            highest_p_state = trellis_bptr[state_idx][highest_p_state]
            state_idx -= 1
        sentence_list.reverse()
        test_labels.append(sentence_list)

    return test_labels