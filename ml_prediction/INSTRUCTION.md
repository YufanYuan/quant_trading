# ML Prediction Strategy - 机器学习预测策略

## 策略概述

这是一个基于深度学习的加密货币交易策略，使用过去2小时（120分钟）的市场数据预测未来30分钟内价格首先达到+0.5%还是-0.5%。

### 核心思路

1. **输入数据**：过去120分钟的OHLC数据及技术指标
2. **特征工程**：计算LogReturn、EMA、RSI等归一化特征
3. **模型预测**：使用LSTM或Transformer模型预测涨跌方向
4. **交易执行**：基于预测结果开仓，设置0.5%的止损和止盈

### 训练数据

- **交易对**：BTCUSDT, DOGEUSDT, ETHUSDT
- **时间范围**：2022年 - 2024年
- **数据粒度**：1分钟K线

### 回测数据

- **时间范围**：2025年
- **目的**：验证策略在未见过的数据上的表现

## 文件结构

```
ml_prediction/
├── __init__.py              # 包初始化
├── features.py              # 特征工程模块
├── label_generator.py       # 标签生成器
├── data_loader.py           # 数据下载和准备
├── model.py                 # 深度学习模型定义
├── train.py                 # 模型训练脚本
├── strategy.py              # 策略执行器
├── backtest.py              # 回测脚本
├── INSTRUCTION.md           # 本文档
└── data/                    # 数据目录（自动创建）
    ├── BTCUSDT_2022_1m.csv
    ├── DOGEUSDT_2023_1m.csv
    ├── ...
    ├── train_features.npy
    └── train_labels.npy
```

## 依赖安装

首先需要安装PyTorch和相关依赖：

```bash
# 安装PyTorch（请根据你的CUDA版本选择合适的命令）
# CPU版本
pip install torch torchvision torchaudio

# CUDA 11.8版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 其他已有依赖（已在pyproject.toml中）
# pandas, numpy, requests等
```

## 使用流程

### 步骤1：下载和准备数据

```bash
cd ml_prediction
python data_loader.py
```

这会：
- 使用项目根目录的`market_data.py`从Binance Vision下载历史数据
- 下载BTCUSDT、DOGEUSDT、ETHUSDT的2022-2024年1分钟数据
- 月度数据会自动缓存到`../data_cache`目录（项目级别的缓存）
- 处理后的年度数据保存到`ml_prediction/data`目录
- 计算特征（OHLC LogReturn, EMA, RSI）
- 生成标签（未来30分钟内首先达到+0.5%还是-0.5%）
- 保存最终训练数据到 `data/train_features.npy` 和 `data/train_labels.npy`

**优势**：
- 使用Binance Vision官方历史数据存档，比实时API更稳定
- 自动缓存机制，避免重复下载
- 与项目其他部分共享数据缓存

### 步骤2：训练模型

```bash
python train.py
```

训练配置（可在 `train.py` 中修改）：
- 模型类型：LSTM或Transformer
- 批次大小：64
- 学习率：0.001
- 验证集比例：20%
- Early Stopping：15个epoch无改善则停止

训练完成后会生成：
- `best_model.pth`：最佳模型权重
- `model_config.json`：模型配置文件

### 步骤3：运行回测

```bash
python backtest.py
```

回测会：
- 加载训练好的模型
- 在2025年数据上进行回测
- 生成交易记录、权益曲线和统计指标
- 保存结果到 `results/` 目录

回测结果包括：
- `results/trades.csv`：所有交易记录
- `results/equity_curve.csv`：权益曲线数据
- `results/statistics.json`：统计指标

## 特征说明

### 输入特征（10维）

对于每个120分钟的窗口，计算以下特征：

1. **OHLC LogReturn**（4维）
   - Open价格的对数收益率（相对于窗口起始价格）
   - Close价格的对数收益率
   - High价格的对数收益率
   - Low价格的对数收益率

2. **EMA LogReturn**（5维）
   - EMA5的对数收益率
   - EMA12的对数收益率
   - EMA26的对数收益率
   - EMA50的对数收益率
   - EMA200的对数收益率

3. **RSI**（1维）
   - 14周期RSI，归一化到0-1范围

### 标签定义

- **0**：未来30分钟内首先达到-0.5%（下跌，做空）
- **1**：未来30分钟内首先达到+0.5%（上涨，做多）
- **2**：30分钟内都未达到（横盘，不交易）

**重要**：横盘样本不是无效样本！模型需要学会识别横盘并选择不交易，这是策略成功的关键。

## 模型架构

### LSTM模型（默认）

```
输入 (batch_size, 120, 10)
  ↓
双向LSTM (hidden_size=128, num_layers=2)
  ↓
全连接层 (128*2 → 64)
  ↓
ReLU + Dropout(0.3)
  ↓
全连接层 (64 → 3)  ← 三分类
  ↓
输出 (batch_size, 3)  [跌, 涨, 横盘]
```

### Transformer模型（可选）

```
输入 (batch_size, 120, 10)
  ↓
输入投影 (10 → 128)
  ↓
位置编码
  ↓
Transformer编码器 (d_model=128, nhead=8, num_layers=3)
  ↓
平均池化
  ↓
全连接层 (128 → 64 → 3)  ← 三分类
  ↓
输出 (batch_size, 3)  [跌, 涨, 横盘]
```

## 交易逻辑

### 开仓条件

1. 当前无持仓
2. 模型预测置信度 ≥ 60%
3. 预测方向为上涨（类别1）或下跌（类别0）
4. **如果预测横盘（类别2），则不交易**

### 平仓条件

1. **止盈**：价格达到目标价位（±0.5%）
2. **止损**：价格触及止损价位（±0.5%）
3. **强制平仓**：回测结束时仍有持仓

### 手续费

- 默认手续费率：0.04%（Binance现货手续费）
- 每次开仓和平仓都会计入手续费

## 性能优化建议

### 数据方面

1. **增加更多交易对**：扩展训练数据多样性
2. **数据平衡**：使用SMOTE等方法平衡正负样本
3. **特征工程**：尝试添加更多技术指标

### 模型方面

1. **超参数调优**：使用网格搜索或贝叶斯优化
2. **集成学习**：训练多个模型并投票
3. **正则化**：调整dropout、weight_decay等参数

### 策略方面

1. **动态阈值**：根据市场波动率调整止损止盈
2. **仓位管理**：根据预测置信度调整仓位大小
3. **过滤机制**：添加市场环境过滤器（如波动率过滤）

## 风险提示

1. **过拟合风险**：模型可能在训练集上表现良好但在新数据上失效
2. **市场变化**：加密货币市场波动大，历史数据可能无法预测未来
3. **滑点和手续费**：实盘交易中的滑点和手续费可能高于回测
4. **技术风险**：网络延迟、API限制等可能影响实盘表现

## 常见问题

### Q: 如何切换到Transformer模型？

A: 在 `train.py` 中修改配置：
```python
config = {
    'model_type': 'transformer',  # 改为 'transformer'
    ...
}
```

### Q: 如何调整预测窗口和阈值？

A: 在 `data_loader.py` 的 `DatasetPreparator` 中修改：
```python
preparator = DatasetPreparator(
    window_size=120,    # 输入窗口（分钟）
    threshold=0.005,    # 涨跌幅阈值（0.5%）
    horizon=30          # 预测窗口（分钟）
)
```

### Q: 训练需要多长时间？

A: 取决于数据量和硬件配置：
- CPU训练：可能需要数小时
- GPU训练（推荐）：通常30分钟-1小时

### Q: 如何使用实盘数据？

A: 修改 `strategy.py` 创建实时数据接口，或使用Backtrader等回测框架集成。

## 下一步改进

- [ ] 添加注意力机制可视化
- [ ] 实现模型解释性分析（SHAP值）
- [ ] 添加实时交易接口
- [ ] 实现多时间框架分析
- [ ] 添加风险管理模块（最大回撤控制）
- [ ] 支持更多交易对和交易所

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [LSTM for Time Series](https://arxiv.org/abs/1506.00019)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## 许可

本策略仅供学习和研究使用，不构成投资建议。使用者需自行承担交易风险。
