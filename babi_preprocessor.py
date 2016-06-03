"""Very basic preprocessing script for converting babi two supporting facts task
data into something Blocks/Theano can handle."""
import os
import json

import h5py
import numpy as np

# I know these things from observation
QLEN = 3
# Length of one statement
SLEN = 3
# I had to find this out by looping over the first 1K questions and getting
# the maximum length of a story
# for training was 61
# for testing was 68
MAX_STORY_LEN = 68


def is_question(line):
    return 'eval' in line


def is_first_statement(token_list):
    return int(token_list[0]) == 1


question_counter = 0
requested_questions = 1000
root_dir = "/media/data/babi-tasks-local"
fname_tpl = "babi-task2-200stories-test.{}"
f_path = os.path.join(root_dir, fname_tpl.format("txt"))
vocab_json_path = os.path.join(root_dir, fname_tpl.format("vocab.json"))
h5path = os.path.join(root_dir, fname_tpl.format("h5"))

# Vocabulary creation
vocab_set = set()
with open(f_path) as text_f, open(vocab_json_path, 'w') as vocab_file:
    for line in text_f:
        if question_counter > requested_questions:
            break

        tokens = line.split()
        if is_question(tokens):
            question_counter += 1
            # this is specific to the task!!
            words = tokens[1:-2]
        else:
            words = tokens[1:]
        for word in words:
            vocab_set.add(word)
    vocab = dict((w, indx + 1) for indx, w in enumerate(vocab_set))
    json.dump(vocab, vocab_file)

# data conversion
question_counter = 0
with open(f_path) as text_f, h5py.File(h5path, 'w') as h5_f:
    questions = h5_f.create_dataset('questions', dtype='l',
                                    shape=(requested_questions, QLEN))
    answers = h5_f.create_dataset('answers', dtype='l',
                                  data=np.zeros(requested_questions))
    stories = h5_f.create_dataset('stories', dtype='l',
                                  shape=(requested_questions, MAX_STORY_LEN, SLEN))

    story_lines = []
    question_indices = []
    prev_story_complete = False
    for line in text_f:

        if question_counter >= requested_questions and prev_story_complete:
            break

        prev_story_complete = False

        tokens = line.split()
        if is_question(tokens):
            q, a = tokens[1:-3], tokens[-3]
            questions[question_counter] = [vocab[w] for w in q]
            # print(questions[question_counter].shape)
            answers[question_counter] = vocab[a]
            question_indices.append(question_counter)
            question_counter += 1

        else:
            if is_first_statement(tokens):
                prev_story_complete = True
                story_ar = np.array([[vocab[w] for w in s] for s in story_lines])
                num_sents = len(story_ar)
                for indx in question_indices:
                    stories[indx, :num_sents] = story_ar
                story_lines = []
                question_indices = []

            story_lines.append(tokens[1:])
