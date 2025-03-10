import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTransformer(nn.Module):
    """
    一个简化的Transformer模型，用于教学演示
    """
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器和解码器
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 参数初始化
        self._reset_parameters()
        
        # 模型维度
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, memory_mask=None):
        """
        模型前向传播
        
        Args:
            src: 源序列 [src_len, batch_size]
            tgt: 目标序列 [tgt_len, batch_size]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            src_padding_mask: 源序列填充掩码
            tgt_padding_mask: 目标序列填充掩码
            memory_mask: 记忆掩码
        
        Returns:
            output: 输出张量
        """
        # 源序列编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask, src_padding_mask)
        
        # 目标序列编码
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        # 解码
        output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask,
                                         tgt_padding_mask, src_padding_mask)
        
        # 生成输出
        output = self.output_layer(output)
        
        return output
    
    def get_attention_weights(self, src, tgt, src_mask=None, tgt_mask=None):
        """获取注意力权重（用于可视化）"""
        # 这里只是一个示例方法，实际实现需要修改PyTorch源码或使用其他方法获取注意力权重
        # 在实际情况下，可能需要修改模型结构或使用钩子函数获取
        batch_size = src.size(1)
        src_len = src.size(0)
        tgt_len = tgt.size(0)
        
        # 生成一个随机注意力权重示例
        attention_weights = torch.rand(self.nhead, batch_size, tgt_len, src_len)
        
        # 应用softmax使权重和为1
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        return attention_weights


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 创建一个用于演示的简单MultiHeadAttention模块
class SimpleMultiHeadAttention(nn.Module):
    """
    简化版多头注意力模块，用于教学演示
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super(SimpleMultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # 定义线性投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, attn_mask=None):
        """
        多头注意力计算
        
        Args:
            query: 查询张量 [seq_len, batch_size, d_model]
            key: 键张量 [seq_len, batch_size, d_model]
            value: 值张量 [seq_len, batch_size, d_model]
            attn_mask: 注意力掩码
        
        Returns:
            output: 注意力输出
            attn_weights: 注意力权重
        """
        # 获取维度
        seq_len, batch_size, _ = query.size()
        
        # 线性投影并分割为多头
        q = self.q_proj(query).view(seq_len, batch_size, self.nhead, self.head_dim).transpose(0, 2)
        k = self.k_proj(key).view(-1, batch_size, self.nhead, self.head_dim).transpose(0, 2)
        v = self.v_proj(value).view(-1, batch_size, self.nhead, self.head_dim).transpose(0, 2)
        
        # 计算注意力得分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（如果有）
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到值
        output = torch.matmul(attn_weights, v)
        
        # 调整输出形状并进行最终的线性变换
        output = output.transpose(0, 2).contiguous().view(seq_len, batch_size, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights 