from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from onmt.encoders.transformer import TransformerEatEncoder, TransformerXEncoder
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
        # Replace the module forward call with a mock. Note that the original module has to be deleted first to bypass the __setattr__ check in nn.Module.
        mocked_almt = Mock()
        mocked_almt.side_effect = lambda x: x
        del encoder.embeddings.almt
        encoder.embeddings.almt = mocked_almt

        src = torch.randint(VOCAB_SIZE, (SEQ_LEN * 3, BATCH_SIZE, 1))
        lengths = (torch.randint(SEQ_LEN, (BATCH_SIZE,)) + 1) * 3
        lengths[0] = SEQ_LEN * 3
        encoder.crosslingual_off()
        encoder.mapping_off()
        _ = encoder(src, lengths)
        mocked_almt.assert_not_called()
        encoder.crosslingual_on()
        encoder.mapping_on()
        _ = encoder(src, lengths)
        mocked_almt.assert_called()

    def test_base(self):
        self._test_routine('base')

    def test_eat(self):
        self._test_routine('eat')
