"""
模型定义模块

实现基于Attention的Transformer模型用于时间序列分类
"""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """
    Transformer分类器

    使用Transformer编码器处理时间序列特征
    """

    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.3
    ):
        """
        Args:
            input_size: 输入特征维度
            seq_len: 序列长度
            d_model: Transformer模型维度
            nhead: 多头注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
        """
        super(TransformerClassifier, self).__init__()

        # 输入投影层
        self.input_projection = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 分类头：1D卷积 + 全连接
        # Conv1d对d_model维度进行卷积，输出1个通道
        # 使用奇数kernel_size便于same padding
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=31,
            padding=15  # same padding: (31 - 1) // 2 = 15
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 全连接层：从seq_len映射到3分类
        self.fc_output = nn.Linear(seq_len, 3)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_size)

        Returns:
            输出logits，形状为 (batch_size, 3)
        """
        # 输入投影
        x = self.input_projection(x)

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码
        x = self.transformer_encoder(x)
        # x: (batch_size, seq_len, d_model)

        # 转置为Conv1d格式: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)

        # 1D卷积: (batch_size, d_model, seq_len) -> (batch_size, 1, seq_len)
        x = self.conv1d(x)
        x = self.relu(x)

        # Squeeze掉通道维度: (batch_size, seq_len)
        x = x.squeeze(1)

        # Dropout
        x = self.dropout(x)

        # 全连接层: (batch_size, seq_len) -> (batch_size, 3)
        x = self.fc_output(x)

        return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def test_models():
    """测试模型"""
    batch_size = 32
    seq_len = 120
    input_size = 10

    # 测试数据
    x = torch.randn(batch_size, seq_len, input_size)

    # 测试Transformer
    print("测试Transformer模型 (三分类，带Conv1D)")
    transformer_model = TransformerClassifier(input_size=input_size, seq_len=seq_len)
    transformer_output = transformer_model(x)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {transformer_output.shape}")
    print(f"  参数量: {sum(p.numel() for p in transformer_model.parameters())}")

    # 测试CrossEntropyLoss匹配
    print("\n测试CrossEntropyLoss匹配:")
    criterion = torch.nn.CrossEntropyLoss()
    labels = torch.randint(0, 3, (batch_size,))  # 标签范围 [0, 2]
    loss = criterion(transformer_output, labels)
    print(f"  模型输出形状: {transformer_output.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  标签范围: [{labels.min()}, {labels.max()}]")
    print(f"  Loss: {loss.item():.4f}")
    print("  [OK] CrossEntropyLoss 和 label 格式匹配！")


if __name__ == '__main__':
    test_models()
