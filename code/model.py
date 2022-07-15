class CommonLitModel(nn.Module):
    
    def __init__(self):
        super(CommonLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(MODEL_NAME)
        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, batch_first=True)
        self.regressor = nn.Linear(self.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        out, _ = self.lstm(outputs['last_hidden_state'], None)
        sequence_output = out[:, -1, :]
        logits = self.regressor(sequence_output)
