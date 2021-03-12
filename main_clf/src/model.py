from torch import nn


class MLP(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_size, out_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(x)
        # x = self.dropout(x)
        output = self.linear(x)
        if not self.training:
            output = self.sigmoid(output)
        return output
