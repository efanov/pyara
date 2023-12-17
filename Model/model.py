"""
Module for where models described
"""
from torch import nn

from config import CFG
import torch


class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock,
              self).__init__()  # super(subclass) - we will inheritance fron superclass of subclass в данном случае можно было прсто написать super()
        self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        if not self.first:
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp = self.conv11(x)
        if not self.first:
            out = self.pre_bn(x)
            out = self.lrelu(out)
        else:
            out = x
        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp
        return out


class MFCCModel(nn.Module):
    def __init__(self):
        super(MFCCModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.model_name = 'CNN_model_ResNet'

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.mp(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.mp(out)
        out = self.block7(out)
        out = self.block8(out)
        out = self.block9(out)
        out = self.bn(out)
        out = self.lrelu(out)
        out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out


class LSTM(nn.Module):
    # define all the layers used in model
    def __init__(self, input_dim=80, hidden_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.model_name = 'LSTM_model'
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            num_layers=CFG.lstm_layers,
                            batch_first=True,
                            bidirectional=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.clf = nn.Linear(hidden_size, 2)
        self._fc = torch.nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=2, bias=True))

    def forward(self, input, length=CFG.width - 1):
        # input: (batch_size, hidden_size, seq_len)
        out, _ = self.lstm(input.transpose(-1, -2))
        out = out[:, CFG.width - 1, :]
        # output: (batch_size, seq_len, hidden_size)

        #         last_hidden = torch.gather(
        #             output,
        #             dim =1,
        #             index = length.sub(1).view(-1, 1, 1).expand(-1, -1, self.hidden_size)
        #         )
        # logits = self.clf(last_hidden.squeeze(dim=1))
        # print(f'OUT SHAPE: {out.shape}')
        out = self._fc(out)
        return out

# # New LSTM
# class LSTM(nn.Module):
#     def __init__(self, input_size = 300 , hidden_size = 128, num_layers = 10, output_size = 1):
#         super(LSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out




