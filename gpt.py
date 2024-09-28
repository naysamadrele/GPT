import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def encode(examples):
    return tokenizer(examples['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=128)

token = dataset.map(encode, batched=True)
dataloader = DataLoader(token, batch_size=8, shuffle=True)

class GPT(nn.Module):
    def _init_(self, vocab, model, head, layer):
        super(GPT, self)._init_()
        self.embedding = nn.Embedding(vocab, model)
        self.transformer_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model, head), layer
        )
        self.fc_out = nn.Linear(model, vocab)
        
    def forward(self, x):
        embedded = self.embedding(x)
        transformer = self.transformer_layer(embedded)
        logits = self.fc_out(transformer)
        return logits

model = 128
head = 4
layer = 4
vocab = tokenizer.vocab

model = GPT(vocab, model, head, layer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 2

for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        ids = batch['ids']
        labels = ids.clone()
        
        optimizer.zero_grad()
        outputs = model(ids)
        loss = criterion(outputs.view(-1, vocab), labels.view(-1))
        
        loss.backward()
        optimizer.step()
    
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

print("Training complete!")
