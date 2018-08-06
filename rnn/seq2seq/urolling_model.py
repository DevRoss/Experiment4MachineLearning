#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Created by Ross on 18-8-6
# Following the tutorial in https://github.com/ematvey/tensorflow-seq2seq-tutorials/blob/master/1-seq2seq.ipynb
import helpers
import tensorflow as tf
import numpy as np
import sys

print(sys.path)
x = [[5, 7, 8], [6, 3], [3], [1]]
xt, xlen = helpers.batch(x)
# print(xt)
# print(xlen)


# 全局变量
PAD = 0
EOS = 1
vocab_size = 10
input_embedding_size = 20
encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='encoder_inputs')
decoder_targets = tf.placeholder(tf.int32, shape=(None, None), name='decoder_targets')
decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name='decoder_inputs')

embeddings = tf.get_variable('embeddings', (vocab_size, input_embedding_size), dtype=tf.float32,
                             initializer=tf.initializers.random_uniform(-1, 1))
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

encoder_cell = tf.nn.rnn_cell.LSTMCell(encoder_hidden_units)
encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                         encoder_inputs_embedded,
                                                         dtype=tf.float32,
                                                         time_major=True)

del encoder_outputs
# print(encoder_final_state)
decoder_cell = tf.nn.rnn_cell.LSTMCell(decoder_hidden_units)
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                         decoder_inputs_embedded,
                                                         initial_state=encoder_final_state,
                                                         dtype=tf.float32,
                                                         time_major=True,
                                                         scope='plain_decoder')
decoder_logits = tf.layers.dense(decoder_outputs,
                                 vocab_size)

decoder_prediction = tf.argmax(decoder_logits, axis=2, name='prediction_layer')
print(decoder_logits)
# print(decoder_predict)

step_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
                                                             logits=decoder_logits)

loss = tf.reduce_mean(step_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

def next_feed():
    batch = next(batches)
    encoder_inputs_, _ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    decoder_inputs_, _ = helpers.batch(
        [[EOS] + (sequence) for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }

if __name__ == '__main__':

    batch_size = 100

    batches = helpers.random_sequences(length_from=3, length_to=8,
                                       vocab_lower=2, vocab_upper=10,
                                       batch_size=batch_size)
    loss_track = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        max_batches = 3001
        batches_in_epoch = 1000

        try:
            for batch in range(max_batches):
                fd = next_feed()
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)

                if batch == 0 or batch % batches_in_epoch == 0:
                    print('batch {}'.format(batch))
                    print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()
        except KeyboardInterrupt:
            print('training interrupted')

