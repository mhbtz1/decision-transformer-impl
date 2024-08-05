import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_size):
        super(ScaledDotProductAttention, self).__init__()
        self.embed_size = embed_size
        self.Q = torch.rand(self.embed_size, self.embed_size)
        self.K = torch.rand(self.embed_size, self.embed_size)
        self.V = torch.rand(self.embed_size, self.embed_size)
    def forward(self, tkns: torch.Tensor, mask=None): # tkns dim: ( embed_size, num_tokens)
        Q = torch.matmul(self.Q, tkns)
        K = torch.matmul(self.K, tkns)
        V = torch.matmul(self.V, tkns)
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.embed_size, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask==0, 1e-9)
        
        attn = F.softmax(scores, dim=-1)
        sftmax = torch.matmul(attn, self.V)
        return attn, sftmax

class MultiheadedAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiheadedAttention, self).__init__()
        self.matrix_dim = embed_size // num_heads
        self.dot_attn = [ScaledDotProductAttention(self.matrix_dim) for _ in range(num_heads)]
        
    def forward(self, tkns: torch.Tensor, mask=None):
        pass



model = ScaledDotProductAttention(256)
attn, sftmax = model(torch.rand(256, 10))
print(sftmax)

