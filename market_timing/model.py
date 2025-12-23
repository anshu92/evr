import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

class MarketTimingClassifier(nn.Module):
    def __init__(self, model_name: str = "t5-small", hidden_dim: int = 512, num_classes: int = 2):
        super().__init__()
        print(f"Loading T5 Encoder: {model_name}...")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        
        # Freeze encoder to save compute/memory (optional, but good for small training sets)
        # Unfreezing the last block is a common strategy
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Classification Head
        # T5-small hidden size is 512
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over the sequence length
        # outputs.last_hidden_state: (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.last_hidden_state
        
        # Masked mean pooling
        # Expand mask to match hidden dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        pooled_output = sum_embeddings / sum_mask
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

    def predict(self, texts: list[str], device='cpu'):
        self.eval()
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
            
        return predictions.cpu().numpy(), probs.cpu().numpy()
















