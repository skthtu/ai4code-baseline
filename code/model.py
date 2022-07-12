import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.LSTM(input_size = 769, hidden_size = 100, num_layers = 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        h0 = torch.randn(1, 100)
        c0 = torch.randn(1, 100)
        x = self.top(x, (h0, c0))
        return x
