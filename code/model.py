import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.lstm = nn.LSTM(input_size = 768, hidden_size = 256, num_layers = 1, batch_first=True)
        self.top = nn.Linear(257, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x, _ = self.lstm(x, None)    
        x = torch.cat((x[:, 0, :], fts), 1) 
        x = self.top(x)
        return x
    
