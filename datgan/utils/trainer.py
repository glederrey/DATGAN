#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trainers for the DATGAN
"""

import tensorflow as tf
from tensorpack import TowerTrainer
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper


class GANTrainerClipping(TowerTrainer):
    """
    GAN Trainer with clipping. (Used with the Standard GAN loss)
    """

    def __init__(self, input, model):
        """
        Initialize object.

        Parameters
        ----------
            input: tensorpack.QueueInput
                Input data for the model
            model: datgan.DATGANModel
                tensorflow model to train
        """

        super().__init__()
        inputs_desc = model.get_inputs_desc()

        # Setup input
        cbs = input.setup(inputs_desc)
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, inputs_desc)
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()

        # Define the training iteration by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            g_min_grad = opt.compute_gradients(model.g_loss, var_list=model.g_vars)
            g_min_grad_clip = [
                (tf.clip_by_value(grad, -5.0, 5.0), var)
                for grad, var in g_min_grad
            ]

            g_min_train_op = opt.apply_gradients(g_min_grad_clip, name='g_op')
            with tf.control_dependencies([g_min_train_op]):
                d_min_grad = opt.compute_gradients(model.d_loss, var_list=model.d_vars)
                d_min_grad_clip = [
                    (tf.clip_by_value(grad, -5.0, 5.0), var)
                    for grad, var in d_min_grad
                ]

                d_min_train_op = opt.apply_gradients(d_min_grad_clip, name='d_op')

        self.train_op = d_min_train_op


class GANTrainer(TowerTrainer):
    """
    Similar GAN trainer as GANTrainerClipping without the clipping of the gradient.
    """

    def __init__(self, input, model):
        """
        Initialize object.

        Parameters
        ----------
            input: tensorpack.QueueInput
                Input data for the model
            model: datgan.DATGANModel
                tensorflow model to train
        """

        super().__init__()
        inputs_desc = model.get_inputs_desc()

        # Setup input
        cbs = input.setup(inputs_desc)
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, inputs_desc)
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()

        # Define the training iteration by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            g_min_grad = opt.compute_gradients(model.g_loss, var_list=model.g_vars)

            g_min_train_op = opt.apply_gradients(g_min_grad, name='g_op')
            with tf.control_dependencies([g_min_train_op]):
                d_min_grad = opt.compute_gradients(model.d_loss, var_list=model.d_vars)

                d_min_train_op = opt.apply_gradients(d_min_grad, name='d_op')

        self.train_op = d_min_train_op


class SeparateGANTrainer(TowerTrainer):
    """
    GAN Trainer that runs separately the training of the generator and the discriminator at different rates
    """

    def __init__(self, input, model, d_period=1, g_period=1):
        """
        Initialize object.

        Parameters
        ----------
            input: tensorpack.QueueInput
                Input data for the model
            model: datgan.DATGANModel
                tensorflow model to train
            d_period: int
                period of each d_opt run
            g_period: int
                period of each g_opt run
        """
        super(SeparateGANTrainer, self).__init__()
        self._d_period = int(d_period)
        self._g_period = int(g_period)
        if not min(d_period, g_period) == 1:
            raise ValueError('The minimum between d_period and g_period must be 1.')

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()
        with tf.name_scope('optimize'):
            self.d_min = opt.minimize(model.d_loss, var_list=model.d_vars, name='d_min')
            self.g_min = opt.minimize(model.g_loss, var_list=model.g_vars, name='g_min')

    def run_step(self):
        """
        Define the training iteration.
        """
        if self.global_step % self._d_period == 0:
            self.hooked_sess.run(self.d_min)
        if self.global_step % self._g_period == 0:
            self.hooked_sess.run(self.g_min)
