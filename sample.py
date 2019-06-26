#  Modified from https://github.com/hunkim/word-rnn-tensorflow

import tensorflow as tf

import os
import pickle

from model import Model

sample_args = {
    'save_dir': 'checkpoints',  # model directory to load stored checkpointed models from
    'num_words': 100,  # 'number of words to sample
    'prime': '',  # prime text, first word model starts with
    'pick': 2,  # 1 = weighted pick, 2 = beam search pick
    'width': 5,  # width of the beam search
    'sample': 1,  # 0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces
    'quiet': True  # suppress printing the prime text (default false)
}


def last_period_index(sentences):
    return len(sentences) - sentences[::-1].index('.')


def first_capital_index(sentences):
    for i, char in enumerate(sentences):
        if char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            return i


async def sample(args: dict):
    with open(os.path.join(args['save_dir'], 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    with open(os.path.join(args['save_dir'], 'words_vocab.pkl'), 'rb') as f:
        words, vocab = pickle.load(f)

    model = Model(saved_args, True)
    with tf.compat.v1.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.compat.v1.train.Saver(tf.global_variables())

        ckpt = tf.train.get_checkpoint_state(args['save_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            text = model.sample(
                sess,
                words,
                vocab,
                args['num_words'],
                args['prime'],
                args['sample'],
                args['pick'],
                args['width'],
                args['quiet']
            )
            return text[first_capital_index(text):last_period_index(text)]
