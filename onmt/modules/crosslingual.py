import torch.nn as nn


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
