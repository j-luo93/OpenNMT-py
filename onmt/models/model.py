""" Onmt NMT Model base class definition """
import torch.nn as nn

from onmt.utils.logging import logger


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, predictor=None):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        # Add a predictor to predict roles if needed.
        if predictor is not None:
            self.predictor = predictor

    def forward(self, src, tgt, lengths, bptt=False, task=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        if task is not None:
            task.set_switches(self)
            logger.debug(f'Running {task}.')

        # Deal with LM tasks.
        if task is not None and task.category == 'lm':
            src_emb = self.encoder.embeddings(src)
            output = self.predictor(src_emb)
            return output, None
        else:
            enc_state, memory_bank, lengths = self.encoder(src, lengths)

            tgt = tgt[:-1]  # exclude last target from inputs
            if bptt is False:
                self.decoder.init_state(src, memory_bank, enc_state)
            dec_out, attns = self.decoder(tgt, memory_bank,
                                          memory_lengths=lengths)
            return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
