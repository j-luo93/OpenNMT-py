import torch
import torch.nn as nn


class EatLayer(nn.Module):

    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.event_hidden_layer = nn.Linear(d_in, d_hid)
        self.agent_hidden_layer = nn.Linear(d_in, d_hid)
        self.theme_hidden_layer = nn.Linear(d_in, d_hid)
        self.hidden = nn.Linear(d_hid * 3, d_out)

    def forward(self, event, agent, theme):
        event_hidden = self.event_hidden_layer(event)
        agent_hidden = self.agent_hidden_layer(agent)
        theme_hidden = self.theme_hidden_layer(theme)
        cat = torch.cat([event_hidden, agent_hidden, theme_hidden], dim=-1)
        output = self.hidden(cat)
        return output
