from unittest import TestCase

import torch
import torch.nn as nn
from onmt.encoders.transformer import TransformerEatEncoder
from onmt.modules.eat_layer import EatLayer
from onmt.modules.embeddings import Embeddings

DIM = 25
SEQ_LEN = 15
BATCH_SIZE = 16
VOCAB_SIZE = 100


class TestEatLayer(TestCase):

    def setUp(self):
        self.eat_layer = EatLayer(DIM, DIM, DIM)

    def test_basic(self):
        e_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        a_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        t_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        output = self.eat_layer(e_in, a_in, t_in)
        self.assertTupleEqual(output.shape, (SEQ_LEN, BATCH_SIZE, DIM))


class TestTransformerEat(TestCase):

    def setUp(self):
        emb = Embeddings(DIM, VOCAB_SIZE, 0)
        self.encoder = TransformerEatEncoder(1, DIM, 5, DIM, 0.1, 0.1, emb, 0)

    def test_basic(self):
        src = torch.randint(VOCAB_SIZE, (SEQ_LEN * 3, BATCH_SIZE, 1))
        lengths = (torch.randint(SEQ_LEN, (BATCH_SIZE,)) + 1) * 3
        lengths[0] = SEQ_LEN * 3
        emb, out, lengths = self.encoder(src, lengths=lengths)
