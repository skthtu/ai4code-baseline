import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
class MarkdownModel(nn.Module):
    
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)
        self.cnn1 = nn.Conv1d(769, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)

    def forward(self,ids, mask, fts):
        #outputs = self.model(ids, mask)
        #last_hidden_state = outputs['last_hidden_state'].permute(0, 2, 1)
        #cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        #cnn_embeddings = self.cnn2(cnn_embeddings)
        #logits, _ = torch.max(cnn_embeddings, 2)
        
        x = self.model(ids, mask)[0] #batchsize*文章の長さ*特徴次元
        x = x.permute(0, 2, 1) #batchsize* 特徴次元 * 文章の長さ
        x = torch.cat((x, fts), 1) 
        x = F.relu(self.cnn1(x))
        x = self.cnn2(x)
        logits, _ = torch.max(x, 2)
        
        return logits
