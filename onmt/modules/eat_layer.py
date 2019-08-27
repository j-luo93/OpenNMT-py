import torch
import torch.nn as nn
import torch.nn.functional as F


class EatLayer(nn.Module):

    def __init__(self, d_in, d_hid, d_out, mode='nonlinear'):
        super().__init__()
        self.event_hidden_layer = nn.Linear(d_in, d_hid)
        self.agent_hidden_layer = nn.Linear(d_in, d_hid)
        self.theme_hidden_layer = nn.Linear(d_in, d_hid)
        self.hidden = nn.Linear(d_hid * 3, d_out)
        assert mode in ['nonlinear', 'sum']
        self.mode = mode

    def forward(self, event, agent, theme):
        if self.mode == 'sum':
            return event + agent + theme

        event_hidden = self.event_hidden_layer(event)
        agent_hidden = self.agent_hidden_layer(agent)
        theme_hidden = self.theme_hidden_layer(theme)
        cat = torch.cat([event_hidden, agent_hidden, theme_hidden], dim=-1)
        output = self.hidden(F.leaky_relu(cat, negative_slope=0.1))
        return output
