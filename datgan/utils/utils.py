#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Class used for training the DATGAN with the WGAN loss
"""

import tensorflow as tf
from tensorpack import Callback
from tensorpack.utils import logger


class ClipCallback(Callback):
    """
    Callback class that clips the value of the gradient usign the WGAN loss

    """
    def _setup_graph(self):

        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            n = v.op.name
            if not n.startswith('discrim/'):
                continue
            logger.info("Clip {}".format(n))
            ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
        self._op = tf.group(*ops, name='clip')

    def _trigger_step(self):
        self._op.run()
