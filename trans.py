import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-9)
        
        attn = F.softmax(scores, dim=-1)
        sftmax = torch.matmul(attn, V)
        return attn, sftmax


model = ScaledDotProductAttention(256)
attn, sftmax = model(torch.rand(32, 32), torch.rand(32, 32), torch.rand(32, 32))
print(sftmax)

