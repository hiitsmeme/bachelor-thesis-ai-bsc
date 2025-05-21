import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rates=[0.5,0.25]):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)

        self.relu = nn.ReLU()
        self.input_dropout = nn.Dropout(dropout_rates[0])
        self.hidden_dropout = nn.Dropout(dropout_rates[1])

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.relu(self.fc1(x))
        x = self.hidden_dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x