class MarkdownModel(nn.Module):
    
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)
        self.cnn1 = nn.Conv1d(768, 256, kernel_size=2, padding=1)
        self.cnn2 = nn.Conv1d(256, 1, kernel_size=2, padding=1)

    def forward(self,ids, mask, fts):
        outputs = self.model(ids, mask)
        last_hidden_state = outputs['last_hidden_state'].permute(0, 2, 1)
        cnn_embeddings = F.relu(self.cnn1(last_hidden_state))
        cnn_embeddings = self.cnn2(cnn_embeddings)
        logits, _ = torch.max(cnn_embeddings, 2)

        return logits
