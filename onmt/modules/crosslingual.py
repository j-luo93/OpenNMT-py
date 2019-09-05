from abc import ABC, abstractmethod

import torch.nn as nn
import torch.nn.functional as F


class SwitchableModule:

    def _try_init_switch(self):
        """Initialize switches. This is not delegated to __init__ to avoid MRO mess."""
        if not hasattr(self, '_switches'):
            self._switches = dict()

    def add_switch(self, name, mod_dict, true_switch_name, false_switch_name):
        """Add a switch.

        Args:
            name (str): the name of the switch
            mod_dict (ModuleDict): the ModuleDict object to switch on
            true_switch_name (str): the name of the True-flagged module in the provided `mod_dict`
            false_switch_name (str): similar to `true_switch_name` but False-flagged
        """
        self._try_init_switch()
        if not isinstance(mod_dict, nn.ModuleDict):
            raise TypeError(f'Expecting ModuleDict but got {type(mod_dict)}')
        if name in self._switches:
            raise NameError(f'Name "{name}" already exists as a switch.')

        if true_switch_name not in mod_dict:
            raise NameError(f'true_switch_name "{true_switch_name}" not found in mod_dict')
        if false_switch_name not in mod_dict:
            raise NameError(f'false_switch_name "{false_switch_name}" not found in mod_dict')

        self._switches[name] = {
            'mod_dict': mod_dict,
            'true_switch_name': true_switch_name,
            'false_switch_name': false_switch_name,
            'switch_value': False
        }

    def get_module(self, name):
        self._try_init_switch()
        switch = self._switches[name]
        switch_value = switch['switch_value']
        flag_name = 'true_switch_name' if switch_value else 'false_switch_name'
        name = switch[flag_name]
        return switch['mod_dict'][name]

    def switch(self, name, flag):
        self._try_init_switch()
        if flag not in [True, False]:
            raise ValueError(f'Expecting boolean flag but got {flag}.')

        switch = self._switches[name]
        switch['switch_value'] = flag


class Task(ABC):

    def _check_format(self, format):
        if format not in ['eat', 'plain']:
            raise ValueError(f'Format "{format}" not supported.')

    def __init__(self, data_path, src_format, tgt_format, name=None, index=None):
        self._check_format(src_format)
        self._check_format(tgt_format)

        self.data_path = data_path
        self.src_format = src_format
        self.tgt_format = tgt_format
        self.name = name
        self.index = index

    @abstractmethod
    def set_switches(self, model):
        pass

    def __repr__(self):
        cls_name = type(self).__name__
        return f'{cls_name}(data_path="{self.data_path}")'


class Eat2PlainMonoTask(Task):

    def __init__(self, data_path, name=None, index=None):
        super().__init__(data_path, 'eat', 'plain', name=name, index=index)

    def set_switches(self, model):
        model.encoder.switch('encoder', False)
        model.encoder.embeddings.switch('embedding', False)
        model.encoder.embeddings.switch('almt', False)
        model.decoder.switch('decoder', False)
        model.decoder.embeddings.switch('embedding', False)
        model.decoder.embeddings.switch('almt', False)
        model.generator.switch('generator', False)


class Eat2PlainAuxMonoTask(Eat2PlainMonoTask):

    def set_switches(self, model):
        model.encoder.switch('encoder', True)
        model.encoder.embeddings.switch('embedding', True)
        model.encoder.embeddings.switch('almt', False)
        model.decoder.switch('decoder', True)
        model.decoder.embeddings.switch('embedding', True)
        model.decoder.embeddings.switch('almt', False)
        model.generator.switch('generator', True)


class Eat2PlainCrosslingualTask(Eat2PlainMonoTask):

    def set_switches(self, model):
        model.encoder.switch('encoder', True)
        model.encoder.embeddings.switch('embedding', True)
        model.encoder.embeddings.switch('almt', True)
        model.decoder.switch('decoder', True)
        model.decoder.embeddings.switch('embedding', True)
        model.decoder.embeddings.switch('almt', True)
        model.generator.switch('generator', True)


class MLP(nn.Module):

    def __init__(self, d_in, d_hids, d_out):
        super().__init__()
        self.layers = nn.ModuleList()
        for di, do in zip([d_in] + d_hids, d_hids + [d_out]):
            self.layers.append(nn.Linear(di, do))

    def forward(self, inp):
        for layer in self.layers[:-1]:
            out = layer(inp)
            inp = F.leaky_relu(out. negative_slope=0.1)
        return self.layers[-1](inp)


class RolePredictor(nn.Module):
    """A module that predicts an agent or theme based on the event."""

    def __init__(self, d_emb, vocab_size):
        super().__init__()
        self.agent_pred = MLP(d_emb, [d_emb], vocab_size)
        self.theme_pred = MLP(d_emb, [d_emb], vocab_size)

    def forward(self, inp):
        return self.agent_pred(inp), self.theme_pred(inp)
