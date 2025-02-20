import torch.nn as nn
import torch
import math

#TODO1
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size = x.shape[0]
        token_len = x.shape[1]
        
        query = self.query(x).view(batch_size, token_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = self.key(x).view(batch_size, token_len, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        value = self.value(x).view(batch_size, token_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # print("query: ", query.shape, "key: ", key.shape, "value: ", value.shape)

        attn_prob = nn.functional.softmax(torch.matmul(query, key) / (self.head_dim ** 0.5), dim=3)
        # print("attn_prob: ", attn_prob.shape)
        attn_prob = self.attn_drop(attn_prob)
        
        # output = torch.matmul(attn_prob, value).permute(0, 2, 1, 3).reshape((batch_size, token_len, self.num_heads * self.head_dim))
        output = torch.matmul(attn_prob, value).permute(0, 2, 1, 3).contiguous().view(batch_size, token_len, self.num_heads * self.head_dim)
        output = self.out(output)
        
        return output
        raise Exception('TODO1!')

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    