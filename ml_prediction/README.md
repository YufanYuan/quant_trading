# ML Prediction Strategy

基于深度学习的加密货币交易策略，使用过去2小时的分钟级数据预测未来30分钟内价格走向。

## 快速开始

### 1. 安装依赖

```bash
# 安装PyTorch（根据你的系统选择）
pip install torch  # CPU版本

# 或者 CUDA版本
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 其他依赖已在项目的pyproject.toml中
```

### 2. 下载和准备训练数据

```bash
cd ml_prediction
python data_loader.py
```

这会：
- 使用项目的`market_data.py`从Binance Vision下载2022-2024年的历史数据
- 自动缓存月度数据到`../data_cache`目录（与项目共享缓存）
- 计算特征并生成标签
- 保存训练数据到`ml_prediction/data`目录

**优势**：
- 使用Binance Vision官方历史数据存档，比实时API更稳定可靠
- 自动缓存机制，避免重复下载
- 与项目其他部分共享数据缓存

### 3. 训练模型

```bash
python train.py
```

训练完成后会生成 `best_model.pth` 和 `model_config.json`。

### 4. 运行回测

```bash
python backtest.py
```

回测结果会保存在 `results/` 目录。

## 策略特点

- **输入**：120分钟（2小时）的OHLC数据
- **特征**：10维（OHLC LogReturn + 5个EMA + RSI）
- **预测**：未来30分钟内先涨0.5%、先跌0.5%，或两者都不达到（横盘）
- **三分类**：
  - 0 = 先跌0.5%（做空）
  - 1 = 先涨0.5%（做多）
  - 2 = 横盘（不交易）
- **模型**：LSTM（默认）或Transformer
- **止损止盈**：±0.5%

## 项目结构

```
ml_prediction/
├── features.py          # 特征工程
├── label_generator.py   # 标签生成
├── data_loader.py       # 数据下载
├── model.py            # 模型定义
├── train.py            # 训练脚本
├── strategy.py         # 策略执行
├── backtest.py         # 回测引擎
└── INSTRUCTION.md      # 详细文档
```

## 主要模块

### 特征工程 (features.py)

计算以下特征：
- OHLC的对数收益率
- EMA5/12/26/50/200的对数收益率
- RSI（归一化到0-1）

### 标签生成 (label_generator.py)

生成三分类标签：
- 0：未来30分钟内先跌0.5%（做空信号）
- 1：未来30分钟内先涨0.5%（做多信号）
- 2：未来30分钟内两者都不达到（横盘，不交易）

这样模型能学会识别市场横盘时不交易，提高策略质量！

### 模型 (model.py)

提供两种模型：
- **LSTM**：双向LSTM + 全连接层
- **Transformer**：Transformer编码器 + 分类头

### 策略执行 (strategy.py)

使用训练好的模型：
1. 实时计算特征
2. 预测涨跌方向
3. 根据置信度开仓
4. 止损止盈管理

## 性能指标

回测会计算以下指标：
- 总交易次数
- 胜率
- 平均盈亏
- 总收益率
- 最大回撤
- 夏普比率

## 配置选项

在 `train.py` 中可以修改：
- 模型类型（LSTM/Transformer）
- 学习率
- 批次大小
- Dropout比率
- Early Stopping参数

在 `data_loader.py` 中可以修改：
- 输入窗口大小（默认120分钟）
- 预测窗口（默认30分钟）
- 涨跌幅阈值（默认0.5%）

## 注意事项

1. **数据需求**：首次运行会下载大量数据，需要稳定的网络连接
2. **计算资源**：建议使用GPU训练，CPU训练可能需要数小时
3. **过拟合风险**：模型可能在历史数据上表现良好但在实盘失效
4. **仅供学习**：本策略仅用于学习和研究，不构成投资建议

## 详细文档

请参阅 [INSTRUCTION.md](INSTRUCTION.md) 获取完整的使用说明和技术细节。

## License

本项目仅供学习和研究使用。
