#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File that describes that DATGAN Synthesizer
"""
import networkx as nx
import tensorflow as tf

from tensorpack.utils import logger
from tensorpack.utils.argtools import memoized
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack import ModelDescBase, InputDesc, FullyConnected, BatchNorm, LayerNorm, Dropout

from datgan.utils.dag import get_in_edges


class DATGANSynthesizer(ModelDescBase):
    """
    Synthesizer for the DATGAN model
    """

    def __init__(self, metadata, dag, batch_size, z_dim, noise, l2norm, learning_rate, num_gen_rnn, num_gen_hidden,
                 num_dis_layers, num_dis_hidden, label_smoothing, loss_function, var_order):
        """
        Constructs all the necessary attributes for the DATGANSynthesizer class.

        Parameters
        ----------
        metadata: dict
            Information about the data.
        dag: networkx.DiGraph
            Directed Acyclic Graph provided by the user.
        batch_size: int
            Size of the batch to feed the model at each step. Defined in the DATGAN class.
        z_dim: int
            Dimension of the noise vector used as an input to the generator. Defined in the DATGAN class.
        noise: float
            Upper bound to the gaussian noise added to with the label smoothing. (only used if label_smoothing is
            set to 'TS' or 'OS') Defined in the DATGAN class.
        l2norm: float
            L2 reguralization coefficient when computing the standard GAN loss. Defined in the DATGAN class.
        learning_rate: float
            Learning rate. Defined in the DATGAN class.
        num_gen_rnn: int
            Size of the hidden units in the LSTM cell. Defined in the DATGAN class.
        num_gen_hidden: int
            Size of the hidden layer used on the output of the generator to act as a convolution. Defined in the DATGAN
            class.
        num_dis_layers: int
            Number of layers for the discriminator. Defined in the DATGAN class.
        num_dis_hidden: int
            Size of the hidden layers in the discriminator. Defined in the DATGAN class.
        label_smoothing: str
            Type of label smoothing. Defined in the DATGAN class.
        loss_function: str
            Name of the loss function to be used. Defined in the DATGAN class.
        var_order: list[str]
            Ordered list for the variables. Used in the Generator.
        """

        self.metadata = metadata
        self.dag = dag
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.label_smoothing = label_smoothing
        self.loss_function = loss_function
        self.var_order = var_order

        # Get the number of sources
        self.source_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        self.n_sources = len(self.source_nodes)

        # Parameter used for the WGGP loss function
        self.lambda_ = 10

        # Undefined variables
        self.g_loss = None
        self.d_loss = None

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                                 Generator                                                  """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def generator(self, z):
        """
        Build generator graph.

        Parameters
        ----------
        z: tensorflow.placeholder_with_default
            Playeholder for the noise used as an input to the Generator

        Returns
        -------
        outputs: list[tensorflow.Tensor]
            Output tensors of the Generator, i.e. the synthetic encoded variables

        Raises
        ------
        ValueError: If any of the elements in self.metadata['details'] has an unsupported value in the `type` key.
        """

        # Compute the in_edges of the dag
        in_edges = get_in_edges(self.dag)

        # Create the NN structure
        with tf.variable_scope('LSTM'):

            # Some variables
            outputs = []  # Treated output ready to post_process
            lstm_outputs = {}  # Resized output (not ready for post-processing yet)
            states = {}
            inputs = {}
            noises = {}

            # Transfor the list of inputs into a dictionnary
            for i, n in enumerate(self.source_nodes):
                noises[n] = z[i]

            # Go through all variables
            for col in self.var_order:

                cell = tf.nn.rnn_cell.LSTMCell(self.num_gen_rnn, name='LSTM_cell')

                ancestors = nx.ancestors(self.dag, col)

                info_ = "\033[91mCreating cell for {} (in-edges: {}; ancestors: {})".format(col, len(in_edges[col]),
                                                                                            len(ancestors))
                logger.info(info_)

                # Get info
                col_info = self.metadata['details'][col]

                # Define the input tensor and input state based on the number of in-edges.
                # No ancestors => corresponds to source nodes
                if len(in_edges[col]) == 0:
                    # Input
                    input = tf.get_variable(name='f0-{}'.format(col), shape=(1, self.num_gen_rnn))
                    input = tf.tile(input, [self.batch_size, 1])
                    # LSTM state
                    state = cell.zero_state(self.batch_size, dtype='float32')
                # Only 1 ancestor => simply get the corresponding values
                elif len(in_edges[col]) == 1:
                    ancestor_col = in_edges[col][0]
                    # Input
                    input = inputs[ancestor_col]
                    # LSTM state
                    state = tf.nn.rnn_cell.LSTMStateTuple(states[ancestor_col], lstm_outputs[ancestor_col])
                # Multiple ancestors => we need to use fully connected layers to compute the inputs
                else:
                    # Go through all in edges to get input, attention and state
                    miLSTM_states = []
                    miLSTM_inputs = []
                    miLSTM_lstm_outputs = []
                    for name in in_edges[col]:
                        miLSTM_inputs.append(inputs[name])
                        miLSTM_states.append(states[name])
                        miLSTM_lstm_outputs.append(lstm_outputs[name])

                    # Concatenate the inputs, attention and states
                    with tf.variable_scope("concat-{}".format(col)):
                        # FC for inputs
                        tmp = tf.concat(miLSTM_inputs, axis=1)
                        input = FullyConnected('FC_inputs', tmp, self.num_gen_rnn, nl=None)

                        # FC for states
                        tmp = tf.concat(miLSTM_states, axis=1)
                        tmp_sta = FullyConnected('FC_states', tmp, self.num_gen_rnn, nl=None)

                        # FC for lstm_outputs
                        tmp = tf.concat(miLSTM_lstm_outputs, axis=1)
                        tmp_out = FullyConnected('FC_h_outputs', tmp, self.num_gen_rnn, nl=None)

                        # Transform states and h_outputs in LSTMStateTuple
                        state = tf.nn.rnn_cell.LSTMStateTuple(tmp_sta, tmp_out)

                # Compute the previous outputs
                ancestor_outputs = []
                for n in self.dag.nodes:
                    if n in ancestors:
                        ancestor_outputs.append(lstm_outputs[n])

                # Compute the noise in function of the number of ancestors
                src_nodes = set(ancestors).intersection(set(self.source_nodes))
                src_nodes = list(src_nodes)

                # 0 sources => Source node
                if len(src_nodes) == 0:
                    noise = noises[col]
                # 1 source => Take the corresponding noise
                elif len(src_nodes) == 1:
                    noise = noises[src_nodes[0]]
                # Multiple sources => we need to use a fully connected layer
                else:
                    src_nodes.sort()
                    str_ = '-'.join(src_nodes)

                    # Check if the noise was already computed
                    if str_ in noises:
                        noise = noises[str_]
                    else:
                        # If not the case, we compute the noise by passing it through a FC
                        with tf.variable_scope(str_):
                            tmp_noises = []
                            for n in src_nodes:
                                tmp_noises.append(noises[n])

                            tmp = tf.concat(tmp_noises, axis=1)
                            noise = FullyConnected('FC_noise', tmp, self.z_dim, nl=None)
                            noises[str_] = noise

                # Define the attention vector and create the LSTM cell
                with tf.variable_scope(col):

                    # Learn the attention vector
                    if len(ancestor_outputs) == 0:
                        attention = tf.zeros(shape=(self.batch_size, self.num_gen_rnn), dtype='float32',
                                             name='att0-{}'.format(col))
                    else:
                        alpha = tf.get_variable("alpha", shape=(len(ancestor_outputs), 1, 1))
                        alpha = tf.nn.softmax(alpha, axis=0, name='softmax-alpha')
                        attention = tf.reduce_sum(tf.stack(ancestor_outputs, axis=0) * alpha, axis=0,
                                                  name='att-{}'.format(col))

                    # Concat the input with the attention vector
                    input = tf.concat([input, noise, attention], axis=1)

                    [new_output, new_input, new_lstm_output, new_state] = self.create_cell(cell, col, col_info, input, state)

                # Add the input to the list of inputs
                inputs[col] = new_input

                # Add the state to the list of states
                states[col] = new_state

                # Add the h_outputs to the list of h_outputs
                lstm_outputs[col] = new_lstm_output

                # Add the list of outputs to the outputs (to be used when post-processing)
                for o in new_output:
                    outputs.append(o)

        return outputs

    def create_cell(self, cell, col, col_info, input, state):
        """
        Create the LSTM cell that is used for each variable in the Generator.

        Parameters
        ----------
        cell: LSTMCell
            LSTM cell defined in tensorflow
        col: str
            Name of the current column/variable
        col_info: dct
            Metadata of the current column/variable
        input: tensorflow.Tensor
            Input tensor
        state: tensorflow.Tensor
            Initial cell state

        Returns
        -------
            outputs: list[tensorflow.Tensor]
                Outputs of the current LSTM cell, i.e. encoded synthetic variable
            next_input: tensorflow.Tensor
                Input for the next LSTM cell
            output: tensorflow.Tensor
                Full output of the LSTM cell
            new_state[0]: tensorflow.Tensor
                New cell state for the next LSTM cell

        Raises
        ------
        ValueError: If any of the elements in self.metadata['details'] has an unsupported value in the `type` key.
        """

        # Use the LSTM cell
        output, new_state = cell(input, state)
        outputs = []

        # Pass the output through a fully connected layer to act as a convolution
        hidden = FullyConnected('FC', output, self.num_gen_hidden, nl=tf.tanh)

        # For cont. var, we need to get the probability and the values
        if col_info['type'] == 'continuous':
            w_val = FullyConnected('FC_val', hidden, col_info['n'], nl=tf.tanh)
            w_prob = FullyConnected('FC_prob', hidden, col_info['n'], nl=tf.nn.softmax)

            # 2 outputs here
            outputs.append(w_val)
            outputs.append(w_prob)

            w = tf.concat([w_val, w_prob], axis=1)
        # For cat. var, we only need the probability
        elif col_info['type'] == 'category':
            w = FullyConnected('FC_prob', hidden, col_info['n'], nl=tf.nn.softmax)
            outputs.append(w)

        else:
            raise ValueError(
                "self.metadata['details'][{}]['type'] must be either `category` or "
                "`continuous`. Instead it was {}.".format(col, col_info['type'])
            )

        next_input = FullyConnected('FC_input', w, self.num_gen_rnn, nl=tf.identity)

        return outputs, next_input, output, new_state[0]

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                               Discriminator                                                """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    @auto_reuse_variable_scope
    def discriminator(self, vecs):
        """
        Build the discriminator. It's a fully connected neural network. Exactly the same as in TGAN. The only difference
        is that Layer Normalization is used instead of Batch Normalization with the 'WGGP' loss.

        Parameters
        ----------
        vecs: list[tensorflow.Tensor]
            List of tensors matching the spec of :meth:`inputs`

        Returns
        -------
        tensorpack.FullyConnected:
            logits

        """
        logits = tf.identity(vecs)
        for i in range(self.num_dis_layers):
            with tf.variable_scope('DISCR_FC_{}'.format(i)):
                if i == 0:
                    logits = FullyConnected(
                        'FC', logits, self.num_dis_hidden, nl=tf.identity,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                    )

                else:
                    logits = FullyConnected('FC', logits, self.num_dis_hidden, nl=tf.identity)

                logits = tf.concat([logits, self.batch_diversity(logits)], axis=1)
                if self.loss_function == 'WGGP':
                    logits = LayerNorm('LN', logits)
                else:
                    logits = BatchNorm('BN', logits, center=True, scale=False)
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return FullyConnected('DISCR_FC_TOP', logits, 1, nl=tf.identity)

    @staticmethod
    def batch_diversity(l, n_kernel=10, kernel_dim=10):
        """
        Return the minibatch discrimination vector as defined by Salimans et al., 2016.


        Parameters
        ----------
        l: tensorflow.Tensor
            Input tensor
        n_kernel: int, default 10
            Number of kernel to use
        kernel_dim: int, default 10
            Dimension of the kernels

        Returns
        -------
        tensorflow.Tensor:
            batch diversity tensor

        """
        M = FullyConnected('FC_DIVERSITY', l, n_kernel * kernel_dim, nl=tf.identity)
        M = tf.reshape(M, [-1, n_kernel, kernel_dim])
        M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
        M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
        diff = tf.exp(-tf.reduce_sum(tf.abs(M1 - M2), axis=3))
        return tf.reduce_sum(diff, axis=0)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Loss functions                                                """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def build_losses(self, list_orig_vars, list_synth_vars):
        """
        Build the loss functions for both the Generator and the discriminator. It is built different depending on the
        choice of the loss function

        Parameters
        ----------
        list_orig_vars: list[tensorflow.Tensor]
            List of tensors for the original variables
        list_synth_vars: list[tensorflow.Tensor]
            List of tensors for the synthetic variables
        """

        # Compute the KL loss for the Generator
        kl = self.kl_div(list_orig_vars, list_synth_vars)

        # Transform list of tensors into a concatenated tensor
        orig_vars = tf.concat(list_orig_vars, axis=1)
        synth_vars = tf.concat(list_synth_vars, axis=1)

        # We need the interpolated values for the WGGP loss
        if self.loss_function == 'WGGP':
            alpha = tf.random_uniform(shape=[self.batch_size, 1], minval=0., maxval=1.)
            interp = list_orig_vars + alpha * (synth_vars - orig_vars)

        with tf.variable_scope('discrim'):
            d_logit_orig = self.discriminator(orig_vars)
            d_logit_synth = self.discriminator(synth_vars)
            if self.loss_function is 'WGGP':
                d_logit_interp = self.discriminator(interp)

        if self.loss_function == 'SGAN':
            self.build_SGAN_loss(d_logit_orig, d_logit_synth, kl)
        elif self.loss_function == 'WGAN':
            self.build_WGAN_loss(d_logit_orig, d_logit_synth, kl)
        elif self.loss_function == 'WGGP':
            self.build_WGGP_loss(d_logit_orig, d_logit_synth, interp, d_logit_interp, kl)

    def build_SGAN_loss(self, d_logit_orig, d_logit_synth, kl_div):
        """
        Define the standard loss function

        Parameters
        ----------
        d_logit_orig: tensorpack.FullyConnected
            Logits for the original encoded variables
        d_logit_synth: tensorpack.FullyConnected
            Logits for the synthetic encoded variables
        kl_div: float
            Value of the KL divergence
        """
        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(d_logit_orig)
            score_fake = tf.sigmoid(d_logit_synth)

            # Compute the loss of the discriminator
            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=d_logit_orig,
                        labels=tf.ones_like(d_logit_orig)) * 0.7 + tf.random_uniform(
                        tf.shape(d_logit_orig),
                        maxval=0.3
                    ),
                    name='loss_orig'
                )

                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logit_synth, labels=tf.zeros_like(d_logit_synth)), name='loss_synth')

                d_pos_acc = tf.reduce_mean(
                    tf.cast(score_real > 0.5, tf.float32), name='accuracy_orig')

                d_neg_acc = tf.reduce_mean(
                    tf.cast(score_fake < 0.5, tf.float32), name='accuracy_synth')

                d_loss = 0.5 * d_loss_pos + 0.5 * d_loss_neg + \
                         tf.contrib.layers.apply_regularization(
                             tf.contrib.layers.l2_regularizer(self.l2norm),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim"))

                self.d_loss = tf.identity(d_loss, name='loss')

            # Compute the loss of the
            with tf.name_scope("gen"):
                g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=d_logit_synth, labels=tf.ones_like(d_logit_synth))) + \
                         tf.contrib.layers.apply_regularization(
                             tf.contrib.layers.l2_regularizer(self.l2norm),
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'gen'))

                g_loss = tf.identity(g_loss, name='loss')
                extra_g = tf.identity(kl_div, name='kl_div')
                self.g_loss = tf.identity(g_loss + extra_g, name='final-g-loss')

            add_moving_summary(g_loss, extra_g, self.g_loss, self.d_loss, d_pos_acc, d_neg_acc, decay=0.)

    def build_WGAN_loss(self, d_logit_orig, d_logit_synth, kl_div):
        """
        Define the standard loss function

        Parameters
        ----------
        d_logit_orig: tensorpack.FullyConnected
            Logits for the original encoded variables
        d_logit_synth: tensorpack.FullyConnected
            Logits for the synthetic encoded variables
        kl_div: float
            Value of the KL divergence
        """

        with tf.name_scope("GAN_loss"):
            self.d_loss = tf.reduce_mean(d_logit_synth - d_logit_orig, name='d_loss')
            self.g_loss = tf.negative(tf.reduce_mean(d_logit_synth), name='g_loss')
            kl = tf.identity(kl_div, name='kl_div')
            add_moving_summary(self.d_loss, self.g_loss, kl)

            self.g_loss = tf.add(self.g_loss, kl)

    def build_WGGP_loss(self, d_logit_orig, d_logit_synth, interp, d_logit_interp, kl_div):
        """
        Define the standard loss function

        Parameters
        ----------
        d_logit_orig: tensorpack.FullyConnected
            Logits for the original encoded variables
        d_logit_synth: tensorpack.FullyConnected
            Logits for the synthetic encoded variables
        interp: tensorflow.Tensor
            Tensor of interpolated values between the original and the synthetic variables
        d_logit_interp: tensorpack.FullyConnected
            Logits for the interpolated data between the original and the synthetic variables
        kl_div: float
            Value of the KL divergence
        """

        with tf.name_scope("GAN_loss"):

            self.d_loss = tf.reduce_mean(d_logit_synth - d_logit_orig, name='d_loss')
            self.g_loss = tf.negative(tf.reduce_mean(d_logit_synth), name='g_loss')

            # the gradient penalty loss
            gradients = tf.gradients(d_logit_interp, interp)[0]
            red_idx = list(range(1, interp.shape.ndims))
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=red_idx))
            gradients_rms = tf.sqrt(tf.reduce_mean(tf.square(slopes)), name='gradient_rms')
            gradient_penalty = tf.reduce_mean(tf.square(slopes - 1), name='gradient_penalty')

            kl = tf.identity(kl_div, name='kl_div')
            add_moving_summary(self.d_loss, self.g_loss, gradient_penalty, gradients_rms, kl)

            self.d_loss = tf.add(self.d_loss, self.lambda_ * gradient_penalty)
            self.g_loss = tf.add(self.g_loss, kl)

    def kl_div(self, list_orig_vars, list_synth_vars):
        """
        Compute the KL divergence on the right variables.

        Parameters
        ----------
        list_orig_vars: list[tensorflow.Tensor]
            List of tensors for the original encoded variables
        list_synth_vars: list[tensorflow.Tensor]
            List of tensors for the synthetic encoded variables

        Returns
        -------
        float:
            Sum of the KL divergences for the right variables

        """
        # KL loss
        kl_div = 0.0
        ptr = 0

        # Go through all variables
        for col_id, col in enumerate(self.var_order):
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'continuous':
                # Skip the value. We only compute the KL on the probability vector
                ptr += 1

            dist = tf.reduce_sum(list_synth_vars[ptr], axis=0)
            dist = dist / tf.reduce_sum(dist)

            real = tf.reduce_sum(list_orig_vars[ptr], axis=0)
            real = real / tf.reduce_sum(real)

            kl_div += self.compute_kl(real, dist)
            ptr += 1

        return kl_div

    @staticmethod
    def compute_kl(real, pred):
        """Compute the Kullback–Leibler divergence

        Parameters
        ----------
        real: tensorflow.Tensor
            Real values.
        pred: tensorflow.Tensor
            Predicted values.

        Returns
        -------
        float:
            Computed divergence for the given values.

        """
        return tf.reduce_sum((tf.log(pred + 1e-4) - tf.log(real + 1e-4)) * pred)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                               General model                                                """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def build_graph(self, *inputs):
        """
        Build the whole graph.

        Parameters
        ----------
        inputs: list[tensorflow.Tensor]
            Inputs from the function self.inputs

        """

        # MULTI NOISE
        z = tf.random_normal([self.n_sources, self.batch_size, self.z_dim], name='z_train')
        z = tf.placeholder_with_default(z, [None, None, self.z_dim], name='z')

        # Create the output for the model
        with tf.variable_scope('gen'):
            list_gen = self.generator(z)

        vecs_output = []
        vecs_fake = []
        vecs_real = []

        ptr = 0

        # Go through all variables
        for col in self.var_order:
            # Get info
            col_info = self.metadata['details'][col]

            if col_info['type'] == 'category':

                # OUTPUT
                vecs_output.append(list_gen[ptr])

                # Synthetic variables
                val = list_gen[ptr]

                # Label smoothing
                if self.label_smoothing == 'TS':
                    noise = tf.random_uniform(tf.shape(val), minval=0, maxval=self.noise)
                    val = (val + noise) / tf.reduce_sum(val + noise, keepdims=True, axis=1)

                vecs_fake.append(val)

                # Original variables
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])

                # Label smoothing
                if self.label_smoothing in ['TS', 'OS']:
                    noise = tf.random_uniform(tf.shape(one_hot), minval=0, maxval=self.noise)
                    one_hot = (one_hot + noise) / tf.reduce_sum(one_hot + noise, keepdims=True, axis=1)

                vecs_real.append(one_hot)
                ptr += 1

            elif col_info['type'] == 'continuous':
                for i in range(2):
                    vecs_output.append(list_gen[ptr])
                    vecs_fake.append(list_gen[ptr])
                    vecs_real.append(inputs[ptr])
                    ptr += 1

        # This weird thing is then used for sampling the generator once it has been trained.
        tf.identity(tf.concat(vecs_output, axis=1), name='output')

        self.build_losses(vecs_real, vecs_fake)
        self.collect_variables()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                                              Other functions                                               """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def inputs(self):
        """
        Return metadata about entry data.

        Returns
        -------
        dict:
            dict of input description used by tensorflow following the metadata.

        Raises
        ------
        ValueError:
            If any of the elements in self.metadata['details'] has an unsupported value in the `type` key.

        """
        inputs = []

        for col in self.var_order:
            col_info = self.metadata['details'][col]
            if col_info['type'] == 'continuous':

                n_modes = col_info['n']

                inputs.append(
                    InputDesc(tf.float32,
                              (self.batch_size, n_modes),
                              'input_{}_value'.format(col))
                )

                inputs.append(
                    InputDesc(tf.float32,
                              (self.batch_size, n_modes),
                              'input_{}_cluster'.format(col)
                              )
                )

            elif col_info['type'] == 'category':
                inputs.append(
                    InputDesc(tf.int32,
                              (self.batch_size, 1),
                              'input_{}'.format(col))
                )

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`continuous`. Instead it was {}.".format(col, col_info['type'])
                )

        return inputs

    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        """
        Assign generator and discriminator variables from their scopes.

        Parameters
        ----------
        g_scope: str
            Scope for the generator.
        d_scope: str
            Scope for the discriminator.

        Raises
        ------
        ValueError:
            If any of the assignments fails or the collections are empty.

        """
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)

        if not (self.g_vars or self.d_vars):
            raise ValueError('There are no variables defined in some of the given scopes')

    @memoized
    def get_optimizer(self):
        """
        Return the optimizer depending on the choice of the loss function.

        If the loss function is:
            - SGAN: Adam
            - WGAN: RMSProp
            - WGGP: Adam

        Parameters have been defined based on the recommendations from the articles defining these losses.

        Returns
        -------
        optimizer:
            A tensorflow optimizer

        """

        if self.loss_function == 'SGAN':
            return tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.9)
        elif self.loss_function == 'WGAN':
            return tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.loss_function == 'WGGP':
            return tf.train.AdamOptimizer(self.learning_rate, beta1=0, beta2=0.9)
