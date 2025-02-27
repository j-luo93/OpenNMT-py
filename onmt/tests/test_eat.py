from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn

from onmt.encoders.transformer import (TransformerEatEncoder,
                                       TransformerXEncoder)
from onmt.modules.crosslingual import SwitchableModule
from onmt.modules.eat_layer import EatLayer
from onmt.modules.embeddings import Embeddings, XEmbeddings

DIM = 25
SEQ_LEN = 15
BATCH_SIZE = 16
VOCAB_SIZE = 100


class TestEatLayer(TestCase):

    def _test_mode(self, mode):
        eat_layer = EatLayer(DIM, DIM, DIM, mode=mode)
        e_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        a_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        t_in = torch.randn(SEQ_LEN, BATCH_SIZE, DIM)
        self.eat_layer = EatLayer(DIM, DIM, DIM)
        output = eat_layer(e_in, a_in, t_in)
        self.assertTupleEqual(output.shape, (SEQ_LEN, BATCH_SIZE, DIM))

    def test_nonlinear(self):
        self._test_mode('nonlinear')

    def test_sum(self):
        self._test_mode('sum')

    def test_residual(self):
        self._test_mode('residual')


class TestTransformerEat(TestCase):

    def _get_encoder(self, use_embedding_proj):
        emb = Embeddings(DIM, VOCAB_SIZE, 0)
        encoder = TransformerEatEncoder(1, DIM, 5, DIM, 0.1, 0.1, emb, 0, use_embedding_proj)
        return encoder

    def _test_routine(self, use_embedding_proj):
        src = torch.randint(VOCAB_SIZE, (SEQ_LEN * 3, BATCH_SIZE, 1))
        lengths = (torch.randint(SEQ_LEN, (BATCH_SIZE,)) + 1) * 3
        lengths[0] = SEQ_LEN * 3
        encoder = self._get_encoder(use_embedding_proj)
        emb, out, lengths = encoder(src, lengths=lengths)

    def test_basic(self):
        self._test_routine(False)

    def test_use_embedding_proj(self):
        self._test_routine(True)


class TestTransformerX(TestCase):

    def _get_encoder(self, mode):
        x_emb = XEmbeddings([DIM, DIM], [VOCAB_SIZE, VOCAB_SIZE * 2], 0)
        args = [1, DIM, 5, DIM, 0.1, 0.1, x_emb, 0]
        if mode == 'eat':
            args.append(False)
        encoder = TransformerXEncoder(*args, mode=mode)
        return encoder

    def _test_routine(self, mode):
        encoder = self._get_encoder(mode)

        src = torch.randint(VOCAB_SIZE, (SEQ_LEN * 3, BATCH_SIZE, 1))
        lengths = (torch.randint(SEQ_LEN, (BATCH_SIZE,)) + 1) * 3
        lengths[0] = SEQ_LEN * 3
        encoder.switch('encoder', False)
        encoder.embeddings.switch('embedding', False)
        encoder.embeddings.switch('almt', False)
        _ = encoder(src, lengths)

        encoder.switch('encoder', True)
        encoder.embeddings.switch('embedding', True)
        encoder.embeddings.switch('almt', True)
        _ = encoder(src, lengths)

    def test_base(self):
        self._test_routine('base')

    def test_eat(self):
        self._test_routine('eat')


class TestSwitchableModule(TestCase):

    def setUp(self):

        class Test(nn.Module, SwitchableModule):

            def __init__(self):
                super().__init__()
                self.mod_dict = nn.ModuleDict()
                self.mod_dict['base'] = nn.Linear(DIM, 20)
                self.mod_dict['crosslingual'] = nn.Linear(DIM, 30)

            def forward(self, inp):
                mod = self.get_module('switch')
                return mod(inp)

        self.mod = Test()
        self.mod.add_switch('switch', self.mod.mod_dict, 'crosslingual', 'base')

    def test_get_module(self):
        inp = torch.randn(BATCH_SIZE, DIM)
        # Test switch on.
        self.mod.switch('switch', True)
        out = self.mod(inp)
        self.assertTupleEqual(out.shape, (BATCH_SIZE, 30))
        # Test switch off.
        self.mod.switch('switch', False)
        out = self.mod(inp)
        self.assertTupleEqual(out.shape, (BATCH_SIZE, 20))
