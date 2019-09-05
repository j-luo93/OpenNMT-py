"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import MultiHeadedAttention
from onmt.modules.crosslingual import SwitchableModule
from onmt.modules.eat_layer import EatLayer
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
from onmt.utils.logging import logger


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def prepare_arg_list(cls, opt, embeddings):
        return [
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions
        ]

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        prepared_arg_list = cls.prepare_arg_list(opt, embeddings)
        return cls(*prepared_arg_list)

    def _get_embedding_seq(self, src):
        emb = self.embeddings(src)
        return emb

    def _get_mask(self, lengths):
        mask = ~sequence_mask(lengths).unsqueeze(1)
        return mask

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self._get_embedding_seq(src)

        out = emb.transpose(0, 1).contiguous()
        mask = self._get_mask(lengths)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)


class TransformerEatEncoder(TransformerEncoder):

    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions, use_embedding_proj):
        super().__init__(num_layers, d_model, heads, d_ff, dropout, attention_dropout,
                         embeddings, max_relative_positions)
        self.eat_layer = EatLayer(d_model, d_model, d_model)
        self.use_embedding_proj = use_embedding_proj
        if self.use_embedding_proj:
            self.embedding_proj = nn.Linear(self.embeddings.embedding_size, d_model)

    @classmethod
    def prepare_arg_list(cls, opt, embeddings):
        prepared_arg_list = super().prepare_arg_list(opt, embeddings)
        prepared_arg_list.append(opt.use_embedding_proj)
        return prepared_arg_list

    def _get_embedding_seq(self, src):
        sl, bs, _ = src.shape
        if sl % 3 != 0:
            raise RuntimeError(f'Length should be divided by 3, but got {sl}.')

        sl = sl // 3
        e_sym, a_sym, t_sym = src.view(sl, 3, bs, 1).unbind(dim=1)
        e_in = self.embeddings(e_sym)
        a_in = self.embeddings(a_sym)
        t_in = self.embeddings(t_sym)

        if self.use_embedding_proj:
            e_in = self.embedding_proj(e_in)
            a_in = self.embedding_proj(a_in)
            t_in = self.embedding_proj(t_in)

        emb = self.eat_layer(e_in, a_in, t_in)
        return emb

    def _get_mask(self, lengths):
        if (lengths % 3 > 0).any().item():
            raise RuntimeError(f'Length should be divided by 3.')
        return super()._get_mask(lengths // 3)


class TransformerXEncoder(nn.Module, SwitchableModule):

    def __init__(self, *args, mode='base', encoders=None, share_encoder=False):
        super().__init__()
        if encoders is None:
            cls = self._get_cls(mode)
            encoders = {'base': cls(*args), 'crosslingual': cls(*args)}
        if not isinstance(encoders, dict):
            raise TypeError(f'Expecting a dict, but got {type(encoders)}')

        # Share the encoder if specified.
        breakpoint() # DEBUG
        if share_encoder:
            logger.info('Sharing encoder in TransformerXEncoder.')
            encoders['crosslingual'] = encoders['base']

        self.encoders = nn.ModuleDict(encoders)
        self.add_switch('encoder', self.encoders, 'crosslingual', 'base')

    @property
    def embeddings(self):
        emb1 = self.encoders['base'].embeddings
        emb2 = self.encoders['crosslingual'].embeddings
        if emb1 is not emb2:
            raise RuntimeError('Both encoders should share the same embedding.')
        return emb1

    @classmethod
    def _get_cls(self, mode):
        assert mode in ['base', 'eat']
        cls = TransformerEncoder if mode == 'base' else TransformerEatEncoder
        return cls

    @classmethod
    def from_opt(cls, opt, embeddings, mode='base'):
        enc_cls = cls._get_cls(mode)
        encoders = {
            'base': enc_cls.from_opt(opt, embeddings),
            'crosslingual': enc_cls.from_opt(opt, embeddings)
        }
        return cls(encoders=encoders, share_encoder=opt.crosslingual_share_encoder)

    def forward(self, src, lengths=None):
        encoder = self.get_module('encoder')
        return encoder(src, lengths=lengths)
