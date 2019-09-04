import torch
import torch.nn as nn

from onmt.modules.crosslingual import SwitchableModule
from onmt.modules.util_class import Cast
from onmt.utils import aeq


class Generator(nn.Module):

    def __init__(self, d_in, d_out, gen_func):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.cast = Cast(torch.float32)
        self.gen_func = gen_func

    def forward(self, inp):
        out = self.proj(inp)
        out = self.cast(out)
        return self.gen_func(out)

    def share_embeddings(self, embeddings):
        self.proj.weight = embeddings.word_lut.weight


class XGenerator(nn.Module, SwitchableModule):

    def __init__(self, d_in, d_outs, gen_func):
        super().__init__()
        aeq(len(d_outs), 2)

        self.generators = nn.ModuleDict()
        self.generators['base'] = Generator(d_in, d_outs[0], gen_func)
        self.generators['crosslingual'] = Generator(d_in, d_outs[1], gen_func)

        self.add_switch('generator', self.generators, 'crosslingual', 'base')

    @property
    def gen_func(self):
        f1 = self.generators['base'].gen_func
        f2 = self.generators['crosslingual'].gen_func
        if f1 is not f2:
            raise RuntimeError('gen_func should be identical.')
        return f1

    def forward(self, inp):
        generator = self.get_module('generator')
        return generator(inp)

    def share_embeddings(self, embeddings):
        for k, generator in self.generators.items():
            generator.share_embeddings(embeddings.embed_layers[k])
