import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import OrderedDict


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# class Recurrent(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(Recurrent, self).__init__()
#         self.Wxbx = nn.Linear(in_size, out_size).cuda()
#         self.Wh = nn.Linear(in_size, out_size, bias=False).cuda()
#
#     def forward(self, inputs, hidden):
#         a = self.Wxbx(inputs)
#         b = self.Wh(hidden)
#         return a + b


# Problem 1
class RNN(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        seq_len,
        batch_size,
        vocab_size,
        num_layers,
        dp_keep_prob,
    ):
        """
        emb_size:     The numvwe of units in the input embeddings
        hidden_size:  The number of hidden units per layer
        seq_len:      The length of the input sequences
        vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        num_layers:   The depth of the stack (i.e. the number of hidden layers at 
                      each time-step)
        dp_keep_prob: The probability of *not* dropping out units in the 
                      non-recurrent connections.
                      Do not apply dropout on recurrent connections.
        """
        super(RNN, self).__init__()

        model = OrderedDict()
        self.embedding = WordEmbedding(emb_size, vocab_size).cuda()
        input_size = emb_size
        for i in range(num_layers):
            model[f"Wx{i}"] = nn.Linear(input_size, hidden_size).cuda()
            model[f"Wh{i}"] = nn.Linear(input_size, hidden_size, bias=False).cuda()
            model["tanh"] = nn.Tanh().cuda()
            # model[f"W{i}"] = nn.Linear(hidden_size, hidden_size).cuda()
            # model[f"D{i}"] = nn.Dropout(1 - dp_keep_prob).cuda()
            input_size = hidden_size
        self.fc = nn.Linear(hidden_size, vocab_size).cuda()
        self.dropout = nn.Dropout(1 - dp_keep_prob).cuda()

        self.model = model
        self.init_weights_uniform()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

    def init_weights_uniform(self):
        for key, layer in self.model.items():
            if key.startswith("W"):
                nn.init.uniform_(layer.weight, -.1, .1)
                if not key.startswith("Wh"):
                    nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc.weight, -.1, .1)
        nn.init.zeros_(self.fc.bias)

    def init_hidden(self):
        """
        This is used for the first mini-batch in an epoch, only.
        """
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()

    def forward(self, inputs, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that 
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)

        Returns:
            - Logits for the softmax over output tokens at every time-step.
                  **Do NOT apply softmax to the outputs!**
                  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
                  this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                  These will be used as the initial hidden states for all the 
                  mini-batches in an epoch, except for the first, where the return 
                  value of self.init_hidden will be used.
                  See the repackage_hiddens function in ptb-lm.py for more details, 
                  if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        timesteps = len(inputs)
        logits = torch.zeros(
            (self.seq_len, self.batch_size, self.vocab_size), requires_grad=True
        ).cuda()
        for ts in range(timesteps):
            ts_input = self.embedding(inputs[ts])
            for i in range(self.num_layers):
                out = self.model[f"Wh{i}"](hidden[i].clone())
                out = out + self.model[f"Wx{i}"](ts_input)
                # out = self.model[f"F{i}"](out)
                # hidden[i] = self.model[f"D{i}"](out)
                hidden[i] = self.model["tanh"](out)
                ts_input = hidden[i].clone()
            logits[ts] = self.dropout(self.fc(ts_input))

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, inputs, hidden, generated_seq_len):
        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                           Note that this can be different than the length used 
                           for training (self.seq_len)
        Returns:
            - Sampled sequences of tokens
                        shape: (generated_seq_len, batch_size)
        """
        samples = []
        orig_seq_len = self.seq_len
        self.seq_len = 1
        seed = inputs.view(1, *inputs.shape)
        for i in range(generated_seq_len):
            samples.append(
                torch.max(nn.Softmax(2)(self.forward(seed, hidden)[0]), 2)[1]
            )
            seed = samples[-1]

        self.seq_len = orig_seq_len
        samples = torch.stack(samples)
        return samples


# Problem 2
class GRU(nn.Module):  # Implement a stacked GRU RNN
    """
    Follow the same instructions as for RNN (above), but use the equations for 
    GRU, not Vanilla RNN.
    """

    def __init__(
        self,
        emb_size,
        hidden_size,
        seq_len,
        batch_size,
        vocab_size,
        num_layers,
        dp_keep_prob,
    ):
        super(GRU, self).__init__()

        model = OrderedDict()
        self.embedding = WordEmbedding(emb_size, vocab_size).cuda()
        input_size = emb_size
        for i in range(num_layers):
            model[f"Wr{i}"] = nn.Linear(input_size, hidden_size).cuda()
            model[f"Ur{i}"] = nn.Linear(input_size, hidden_size, bias=False).cuda()
            model[f"Wz{i}"] = nn.Linear(input_size, hidden_size).cuda()
            model[f"Uz{i}"] = nn.Linear(input_size, hidden_size, bias=False).cuda()
            model[f"Wh{i}"] = nn.Linear(input_size, hidden_size).cuda()
            model[f"Uh{i}"] = nn.Linear(input_size, hidden_size, bias=False).cuda()
            model[f"tanh"] = nn.Tanh().cuda()
            model[f"sigmoid"] = nn.Sigmoid().cuda()
            # model[f"W{i}"] = nn.Linear(hidden_size, hidden_size).cuda()
            # model[f"D{i}"] = nn.Dropout(1 - dp_keep_prob).cuda()
            input_size = hidden_size
        self.fc = nn.Linear(hidden_size, vocab_size).cuda()
        self.dropout = nn.Dropout(1 - dp_keep_prob).cuda()

        self.model = model
        self.init_weights_uniform()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dp_keep_prob = dp_keep_prob

    def init_weights_uniform(self):
        for key, layer in self.model.items():
            if key.startswith("W") or key.startswith("U"):
                nn.init.uniform_(layer.weight, -.1, .1)
                if key.startswith("W"):
                    nn.init.zeros_(layer.bias)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).cuda()

    def forward(self, inputs, hidden):
        timesteps = len(inputs)
        logits = torch.zeros(
            (self.seq_len, self.batch_size, self.vocab_size), requires_grad=True
        ).cuda()
        for ts in range(timesteps):
            ts_input = self.embedding(inputs[ts])
            for i in range(self.num_layers):
                rt = self.model[f"Wr{i}"](ts_input)
                rt = self.model["sigmoid"](rt + self.model[f"Ur{i}"](hidden[i].clone()))
                zt = self.model[f"Wz{i}"](ts_input)
                zt = self.model["sigmoid"](zt + self.model[f"Uz{i}"](hidden[i].clone()))
                ht = self.model[f"Wh{i}"](ts_input)
                ht = self.model["tanh"](
                    ht + self.model[f"Uh{i}"](rt * hidden[i].clone())
                )
                out = (torch.ones(zt.shape).cuda() - zt) * hidden[i].clone() + zt * ht
                # out = self.model[f"F{i}"](out)
                # hidden[i] = self.model[f"D{i}"](out)
                hidden[i] = out
                ts_input = hidden[i].clone()
            logits[ts] = self.dropout(self.fc(ts_input))

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

    def generate(self, inputs, hidden, generated_seq_len):
        samples = 0
        return samples


# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.

We're building a transfomer architecture for next-step prediction tasks, and 
applying it to sequential language modelling. We use a binary "mask" to specify 
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.

The model first encodes inputs using the concatenation of a learned WordEmbedding 
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that 
identifies it's position (i.e. time-step).

These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a 
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections, 
and layer normalization.

The complete model consists of the embeddings, the stacked transformer blocks, 
and a linear layer followed by a softmax.
"""

# This code has been modified from an open-source project, by David Krueger.
# The original license is included below:
# MIT License
#
# Copyright (c) 2018 Alexander Rush
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ----------------------------------------------------------------------------------

# TODO: implement this class
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.d_k = n_units // n_heads
        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units

        # TODO: create/initialize any necessary parameters or layers
        # Note: the only Pytorch modules you are allowed to use are nn.Linear
        # and nn.Dropout

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units, self.d_k)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.

        return  # size: (batch_size, seq_len, self.n_units)


# ----------------------------------------------------------------------------------
# The encodings of elements of the input sequence


class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        # print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, n_units, 2).float() * -(math.log(10000.0) / n_units)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return self.dropout(x)


# ----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](
            x, lambda x: self.self_attn(x, x, x, mask)
        )  # apply the self-attention
        return self.sublayer[1](x, self.feed_forward)  # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """

    def __init__(self, layer, n_blocks):  # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(
            self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1
        )


def make_model(vocab_size, n_blocks=6, n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(
            TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks
        ),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size,
    )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


# ----------------------------------------------------------------------------------
# Data processing


def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# ----------------------------------------------------------------------------------
# Some standard modules


class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """

    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
