import torch.nn as nn
import torch.nn.functional as F

# the hidden layer is RNN for deterministic world and FC for stochastic
# because in stochastic world the network runs a batch of state, that distort the hidden state
HIDDEN = [nn.Linear, nn.GRUCell] 

class DDQNAgent(nn.Module):
    """ Identical to rnn_agent, but does not compute value/probability for each action, only the hidden state. """
    def __init__(self, input_shape, args, reccurent=True):
        nn.Module.__init__(self)
        self.reccurent = reccurent
        self.args = args

        self.fc_i   = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.hidden = HIDDEN[reccurent](args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_o   = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc_i.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x1 = F.relu(self.fc_i(inputs))
        # h = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim)) #TODO check
        x2 = self.hidden(x1, hidden_state) if self.reccurent else F.relu(self.hidden(x1))
        q  = self.fc_o(x2)

        # if reccurent=False, the agent ignores x2 (the hidden state)
        return q, x2
