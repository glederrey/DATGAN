#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File that describes the Generator for the DATGAN
"""

import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers

from datgan.utils.dag import get_in_edges


class Generator(tf.keras.Model):
    """
    Generator of the DATGAN. It's using LSTM cells following the DAG provided as a input parameter.
    """

    def __init__(self, metadata, dag, batch_size, z_dim, num_gen_rnn, num_gen_hidden, var_order, loss_function,
                 l2_reg, verbose):
        """
        Initialize the class

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
        num_gen_rnn: int
            Size of the hidden units in the LSTM cell. Defined in the DATGAN class.
        num_gen_hidden: int
            Size of the hidden layer used on the output of the generator to act as a convolution. Defined in the DATGAN
            class.
        var_order: list[str]
            Ordered list for the variables. Used in the Generator.
        loss_function: str
            Name of the loss function to be used. (Defined in the class DATGAN)
        l2_reg: bool
            Use l2 regularization or not
        verbose: int
            Level of verbose
        """
        super().__init__()
        self.metadata = metadata
        self.dag = dag
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_hidden = num_gen_hidden
        self.var_order = var_order
        self.loss_function = loss_function
        self.l2_reg = l2_reg
        self.verbose = verbose

        # Get the number of sources
        self.source_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        self.n_sources = len(self.source_nodes)

        # Stuff done with networkx
        self.in_edges = get_in_edges(self.dag)
        self.ancestors = {}
        self.n_successors = {}
        for col in self.var_order:
            self.ancestors[col] = nx.ancestors(self.dag, col)
            self.n_successors[col] = len(list(self.dag.successors(col)))

        # Regularizer
        if self.l2_reg:
            self.kern_reg = tf.keras.regularizers.L2(1e-5)
        else:
            self.kern_reg = None

        # Parameters
        self.zero_inputs = None
        self.zero_alphas = None

        # Other variables (usually 0)
        self.zero_cell_state = tf.zeros([self.batch_size, self.num_gen_rnn])
        self.zero_hidden_state = tf.zeros([self.batch_size, self.num_gen_rnn])
        self.zero_attention = tf.zeros([self.batch_size, self.num_gen_rnn])

        # LSTM cell
        self.lstms = None

        # Multi input
        self.miLSTM_inputs_layers = None
        self.miLSTM_cell_states_layers = None
        self.miLSTM_hidden_states_layers = None

        # Noise
        self.noise_layers = None

        # Inside the LSTM cell
        self.hidden_layers = None
        self.output_layers = None
        self.input_layers = None

        self.define_parameters()

    def define_parameters(self):
        """
        Define the parameters for the Generator
        """
        # Parameters
        self.zero_inputs = {}
        self.zero_alphas = {}

        # LSTM cell
        self.lstms = {}

        # Multi input
        self.miLSTM_inputs_layers = {}
        self.miLSTM_cell_states_layers = {}
        self.miLSTM_hidden_states_layers = {}

        # Noise
        self.noise_layers = {}

        # Inside the LSTM cell
        self.hidden_layers = {}
        self.output_layers = {}
        self.input_layers = {}

        # Compute the in_edges of the dag
        in_edges = get_in_edges(self.dag)

        # Compute the existing noises (comes from source nodes)
        existing_noises = []
        for i in self.source_nodes:
            existing_noises.append(i)

        for col in self.var_order:
            # We need one lstm cell per variable
            self.lstms[col] = layers.LSTM(self.num_gen_rnn,
                                          return_state=True,
                                          time_major=True,
                                          kernel_regularizer=self.kern_reg,
                                          name='LSTM_{}'.format(col))

            # Get the ancestors of the current variable in the DAG
            ancestors = self.ancestors[col]

            # Get info
            col_info = self.metadata['details'][col]

            # For the input tensor, cell state, and hidden state:

            if len(in_edges[col]) == 0:
                self.zero_inputs[col] = tf.Variable(tf.zeros([1, self.num_gen_rnn]), name='input_{}'.format(col))
            # If the current variable has more than 1 ancestors, we need Linear layers that will be used after
            # concatenating the different inputs, cell states, and hidden states.
            if len(in_edges[col]) > 1:
                self.miLSTM_inputs_layers[col] = layers.Dense(self.num_gen_rnn,
                                                              kernel_regularizer=self.kern_reg,
                                                              name='multi_input_{}'.format(col))
                self.miLSTM_cell_states_layers[col] = layers.Dense(self.num_gen_rnn,
                                                                   kernel_regularizer=self.kern_reg,
                                                                   name='multi_cell_states_{}'.format(col))
                self.miLSTM_hidden_states_layers[col] = layers.Dense(self.num_gen_rnn,
                                                                     kernel_regularizer=self.kern_reg,
                                                                     name='multi_hidden_states_{}'.format(col))

            # Compute the noise in function of the number of ancestors
            src_nodes = set(ancestors).intersection(set(self.source_nodes))
            src_nodes = list(src_nodes)

            # If a variable has multiple source nodes in their ancestors, we need to concatenate these noises and pass
            # them through a Linear layer.
            if len(src_nodes) > 1:
                src_nodes.sort()
                str_ = '-'.join(src_nodes)

                if str_ not in existing_noises:
                    self.noise_layers[str_] = layers.Dense(self.z_dim,
                                                           kernel_regularizer=self.kern_reg,
                                                           name='noise_{}'.format(str_))

                    existing_noises.append(str_)

            # For the attention vector, we have two cases. If there are no ancestors, the attention vector is just
            # initialized as a zero vector.
            if len(ancestors) > 0:
                # If the current variable has at least one ancestor, we are learning the alpha vector instead.
                self.zero_alphas[col] = tf.Variable(tf.zeros([len(ancestors), 1, 1]), name="alpha_{}".format(col))

            # For the cell itself, we have to define multiple layers depending on the type of variables
            self.hidden_layers[col] = layers.Dense(self.num_gen_hidden,
                                                   activation='tanh',
                                                   kernel_regularizer=self.kern_reg,
                                                   name='hidden_layer_{}'.format(col))

            if col_info['type'] == 'continuous':
                self.output_layers[col + '_val'] = layers.Dense(col_info['n'],
                                                                activation='tanh',
                                                                kernel_regularizer=self.kern_reg,
                                                                name='output_cont_val_{}'.format(col))
                self.output_layers[col + '_prob'] = layers.Dense(col_info['n'],
                                                                 activation='softmax',
                                                                 kernel_regularizer=self.kern_reg,
                                                                 name='output_cont_prob_{}'.format(col))

            elif col_info['type'] == 'categorical':
                self.output_layers[col] = layers.Dense(col_info['n'],
                                                       activation='softmax',
                                                       kernel_regularizer=self.kern_reg,
                                                       name='output_cat_{}'.format(col))

            # If there is a successor in the graph, then we need the next input layer
            if self.n_successors[col] > 0:
                self.input_layers[col] = layers.Dense(self.num_gen_rnn,
                                                      kernel_regularizer=self.kern_reg,
                                                      name='next_input_{}'.format(col))

    def call(self, z):
        """
        Build the Generator

        Parameters
        ----------
        z: torch.Tensor
            Noise used as an input for the Generator

        Returns
        -------
        outputs: list[torch.Tensor]
            Output tensors of the Generator, i.e. the synthetic encoded variables

        Raises
        ------
        ValueError: If any of the elements in self.metadata['details'] has an unsupported value in the `type` key.
        """

        # Some variables
        outputs = {} # Encoded synthetic variables that will be returned
        lstm_outputs = {}
        hidden_states = {}
        cell_states = {}
        inputs = {}
        noises = {}

        # Transform the list of inputs into a dictionary
        for i, n in enumerate(self.source_nodes):
            noises[n] = z[i]

        for col in self.var_order:

            # Get the ancestors of the current variable in the DAG
            ancestors = self.ancestors[col]

            # Get info
            col_info = self.metadata['details'][col]

            # Define the input tensor, cell state and hidden state based on the number of in-edges.
            # No ancestors => corresponds to source nodes
            if len(self.in_edges[col]) == 0:
                # Input
                input_ = self.zero_inputs[col]
                input_ = tf.tile(input_, [self.batch_size, 1])
                # Cell state
                cell_state = self.zero_cell_state
                # Hidden state
                hidden_state = self.zero_hidden_state
            # Only 1 ancestor => simply get the corresponding values
            elif len(self.in_edges[col]) == 1:
                ancestor_col = self.in_edges[col][0]
                # Input
                input_ = inputs[ancestor_col]
                # Cell state
                cell_state = cell_states[ancestor_col]
                # Hidden state
                hidden_state = hidden_states[ancestor_col]
            # Multiple ancestors => we need to use fully connected layers to compute the inputs
            else:
                # Go through all in edges to get input, attention and state
                miLSTM_inputs = []
                miLSTM_cell_states = []
                miLSTM_hidden_states = []

                for name in self.in_edges[col]:
                    miLSTM_inputs.append(inputs[name])
                    miLSTM_cell_states.append(cell_states[name])
                    miLSTM_hidden_states.append(hidden_states[name])

                input_ = self.miLSTM_inputs_layers[col](tf.concat(miLSTM_inputs, axis=1))
                cell_state = self.miLSTM_cell_states_layers[col](tf.concat(miLSTM_cell_states, axis=1))
                hidden_state = self.miLSTM_hidden_states_layers[col](tf.concat(miLSTM_hidden_states, axis=1))

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
                    # If not the case, we compute the noise by passing it through a Fully connected layer
                    tmp_noises = []
                    for n in src_nodes:
                        tmp_noises.append(noises[n])

                    noise = self.noise_layers[str_](tf.concat(tmp_noises, axis=1))
                    noises[str_] = noise

            # Get the outputs of the ancestors in the DAG
            ancestor_outputs = []
            for n in ancestors:
                ancestor_outputs.append(lstm_outputs[n])

            # Compute the attention vector
            if len(ancestor_outputs) == 0:
                attention = self.zero_attention
            else:
                alpha = self.zero_alphas[col]
                alpha = tf.nn.softmax(alpha, axis=0)
                attention = tf.reduce_sum(tf.stack(ancestor_outputs, axis=0) * alpha, axis=0)

            # Concatenate the input with the attention vector
            input_ = tf.concat([input_, noise, attention], axis=1)

            [out, next_input, lstm_output, new_cell_state, new_hidden_state] = self.create_cell(col,
                                                                                                col_info,
                                                                                                input_,
                                                                                                cell_state,
                                                                                                hidden_state)

            # Add the input to the list of inputs
            inputs[col] = next_input

            # Add the cell state to the list of cell states
            cell_states[col] = new_cell_state

            # Add the hidden state to the list of hidden states
            hidden_states[col] = new_hidden_state

            # Add the h_outputs to the list of h_outputs
            lstm_outputs[col] = lstm_output

            # Add the list of outputs to the outputs (to be used when post-processing)
            outputs[col] = out

        return outputs

    def create_cell(self, col, col_info, input_, cell_state, hidden_state):
        """
        Create the LSTM cell that is used for each variable in the Generator.

        Parameters
        ----------
        col: str
            Name of the current column/variable
        col_info: dct
            Metadata of the current column/variable
        input_: pytorch.Tensor
            Input tensor
        cell_state: pytorch.Tensor
            Cell state tensor
        hidden_state: pytorch.Tensor
            Hidden state tensor

        Returns
        -------
        outputs: list[pytorch.Tensor]
            Outputs of the current LSTM cell, i.e. encoded synthetic variable
        next_input: pytorch.Tensor
            Input for the next LSTM cell
        lstm_output: pytorch.Tensor
            Output of the LSTM cell
        next_cell_state: pytorch.Tensor
            Cell state for the next LSTM cell
        next_hidden_state: pytorch.Tensor
            Hidden state for the next LSTM cell
        Raises
        ------
            ValueError: If any of the elements in self.metadata['details'] has an unsupported value in the `type` key.
        """
        input_ = tf.expand_dims(input_, axis=0)

        # Use the LSTM cell
        lstm_output, new_hidden_state, new_cell_state = self.lstms[col](input_, initial_state=[hidden_state, cell_state])

        # Pass the output through a fully connected layer to act as a convolution
        hidden = self.hidden_layers[col](lstm_output)

        # For cont. var, we need to get the probability and the values
        if col_info['type'] == 'continuous':

            w_prob = self.output_layers[col + '_prob'](hidden)
            w_val = self.output_layers[col + '_val'](hidden)

            # 2 outputs here
            w = tf.concat([w_val, w_prob], axis=1)

        # For cat. var, we only need the probability
        elif col_info['type'] == 'categorical':
            w = self.output_layers[col](hidden)

        else:
            raise ValueError(
                "self.metadata['details'][{}]['type'] must be either `categorical` or "
                "`continuous`. Instead it was {}.".format(col, col_info['type'])
            )

        if self.n_successors[col] > 0:
            next_input = self.input_layers[col](w)
        else:
            next_input = None

        return w, next_input, lstm_output, new_cell_state, new_hidden_state


