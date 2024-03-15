import torch
import torch.nn as nn

class RNN(nn.Module):
    """
    Vanilla RNN - Language Model
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        # h = g(W1 dot [hidden_t-1, x_t] + b1)
        self.i2h = nn.Linear(in_features = input_size + hidden_size, out_features = hidden_size)
        # y = g(W2 dot hidden_t + b2)
        self.i2o = nn.Linear(in_features = hidden_size, out_features = output_size)
        # tanh activation
        self.tanh = nn.Tanh()
        # softmax activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, prev_hidden):
        # Concat previous hidden state and input
        prev_hidden_and_input = torch.cat(tensors=(prev_hidden, input), dim=1)
        # Calculate the current hidden
        current_hidden = self.tanh(self.i2h(prev_hidden_and_input))
        # Calculate the output
        output = self.softmax(self.i2o(current_hidden))

        return output, current_hidden
    
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
