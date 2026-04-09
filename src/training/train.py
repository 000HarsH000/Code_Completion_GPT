import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CodeDataset(Dataset):
    def __init__(self, bin_file, block_size=1024):
        self.block_size = block_size
        
        # Load tokenized data
        self.data = np.fromfile(bin_file, dtype=np.uint16)
        
        # Total number of training samples
        self.n_samples = len(self.data) - block_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx : idx + self.block_size]
        
        # Target sequence (shifted by 1)
        y = self.data[idx + 1 : idx + self.block_size + 1]
        
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long)
        )


BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
TRAIN_BIN = os.path.join(BASE_DIR, "Data", "tokenized", "train.bin")

train_dataset = CodeDataset(TRAIN_BIN, block_size=1024)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape  # Batch, Time
        
        token_emb = self.token_embedding(x)  # (B, T, C)
        
        positions = torch.arange(T, device=x.device)  # (T,)
        pos_emb = self.position_embedding(positions)  # (T, C)
        x = token_emb + pos_emb
        
        return self.dropout(x)
    
   

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Register causal mask ONCE (not every forward pass)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, h, T, T)

        # Apply causal mask
        mask = self.mask[:T, :T]  # (T, T)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))

        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Attention output
        out = att @ v  # (B, h, T, d)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection
        out = self.resid_dropout(self.out(out))
        
        return out

    
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,block_size):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, block_size)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, block_size=1024, num_heads=8, num_layers=6):
        super().__init__()
        
        self.embedding = GPTEmbedding(vocab_size, embed_dim, block_size)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.embedding.token_embedding.weight

    def forward(self, x, targets=None):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.view(B*T)
            
            loss = nn.CrossEntropyLoss()(logits, targets)
        
        return logits, loss
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(vocab_size=50000).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

def train_step(model, batch, optimizer, device):
    model.train()
    
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    # 🔥 Forward pass
    logits, loss = model(x, y)
    
    # 🔥 Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # 🔥 Optimizer step
    optimizer.step()
    
    return loss.item()


max_steps = 15000  
step_count = 0
total_loss = 0

for epoch in range(1000):  # dummy loop (won't actually reach 1000)
    for step, batch in enumerate(train_loader):
        
        loss = train_step(model, batch, optimizer, device)
        
        total_loss += loss
        
        if step_count % 100 == 0:
            avg_loss = total_loss / (step_count + 1)
            print(f"Step {step_count} | Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}")
        
        step_count += 1
        
        # Stop condition
        if step_count >= max_steps:
            break
    
    if step_count >= max_steps:
        break