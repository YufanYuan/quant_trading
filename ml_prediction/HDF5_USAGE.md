# HDF5数据使用指南

## 重构完成概览

已成功重构数据处理流程，解决了内存问题并优化了数据存储。

### 主要改进

1. **使用pandas优化特征计算**：EMA和RSI使用`ewm()`和`rolling()`方法，效率更高
2. **支持跨年warm-up数据**：处理2023年数据时自动加载2022年末尾200条数据，确保指标连续性
3. **HDF5增量存储**：每处理完一年立即写入，大幅降低内存占用
4. **Features和Labels合并**：存储在单个数组中，shape为`(N, 120, 11)`
5. **NaN检测和清理**：自动检测并丢弃包含NaN的样本
6. **使用float32**：节省50%存储空间

## 数据生成

运行主脚本生成训练数据：

```bash
cd ml_prediction
uv run python data_loader.py
```

这将生成`./data/all_data.h5`文件，包含所有交易对和年份的数据。

## 数据结构

```
all_data.h5
├── BTCUSDT_2022    [shape: (N1, 120, 11), dtype: float32]
├── BTCUSDT_2023    [shape: (N2, 120, 11), dtype: float32]
├── BTCUSDT_2024    [shape: (N3, 120, 11), dtype: float32]
├── ETHUSDT_2022    [shape: (M1, 120, 11), dtype: float32]
└── ...

每个数组的维度：
- 第1维: 样本数量
- 第2维: 时间步（120分钟 = 2小时）
- 第3维: 特征数（10个特征 + 1个label）

特征列（前10列）：
0. open_logret
1. close_logret
2. high_logret
3. low_logret
4. ema5_logret
5. ema12_logret
6. ema26_logret
7. ema50_logret
8. ema200_logret
9. rsi

Label列（最后1列，存储在[:, 0, -1]位置）：
10. label (0=横盘, 1=上涨, 2=下跌)
```

## 数据读取示例

### 1. 读取单个dataset

```python
import h5py
import numpy as np

# 读取BTCUSDT 2024年数据
with h5py.File('./data/all_data.h5', 'r') as f:
    data = f['BTCUSDT_2024'][:]  # shape: (N, 120, 11)

    # 分离features和labels
    features = data[:, :, :-1]  # (N, 120, 10)
    labels = data[:, 0, -1]      # (N,) 注意：只取第一个时间步

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label distribution: {np.bincount(labels.astype(int))}")
```

### 2. 读取多个datasets

```python
import h5py
import numpy as np

datasets_to_load = ['BTCUSDT_2022', 'BTCUSDT_2023', 'ETHUSDT_2022']
all_features = []
all_labels = []

with h5py.File('./data/all_data.h5', 'r') as f:
    for dataset_name in datasets_to_load:
        if dataset_name in f:
            data = f[dataset_name][:]
            all_features.append(data[:, :, :-1])
            all_labels.append(data[:, 0, -1])

# 合并所有数据
features = np.concatenate(all_features, axis=0)
labels = np.concatenate(all_labels, axis=0)

print(f"Total samples: {len(labels)}")
```

### 3. 分批读取（大数据集）

```python
import h5py

with h5py.File('./data/all_data.h5', 'r') as f:
    dataset = f['BTCUSDT_2024']

    # 分批读取，避免内存溢出
    batch_size = 10000
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        features = batch[:, :, :-1]
        labels = batch[:, 0, -1]

        # 处理这个batch...
        print(f"Processing batch {i//batch_size}: {len(batch)} samples")
```

### 4. 推荐的PyTorch DataLoader（分段随机采样 + 自动划分训练/验证集）

**推荐使用 `hdf5_dataloader.py` 提供的DataLoader**，它实现了高效的分段随机采样和自动训练/验证集划分：

```python
from hdf5_dataloader import create_train_dataloader, create_val_dataloader

# 创建训练DataLoader
# 自动从每个dataset的前80%作为训练集
train_loader = create_train_dataloader(
    h5_path='./data/all_data.h5',
    dataset_names=None,  # None表示使用所有datasets
    batch_size=256,
    segments_per_batch=4,  # 每个batch由4个segments组成
    val_ratio=0.2,  # 从每个dataset末尾取20%作为验证集
    num_workers=2,
    seed=42
)

# 创建验证DataLoader
# 自动从每个dataset的后20%作为验证集
val_loader = create_val_dataloader(
    h5_path='./data/all_data.h5',
    dataset_names=None,
    batch_size=256,
    segments_per_batch=4,
    val_ratio=0.2,  # 必须与训练集保持一致
    num_workers=2,
    seed=123
)

# 训练循环
for epoch in range(num_epochs):
    # 训练
    for i, (features, labels) in enumerate(train_loader):
        # features: (256, 120, 10)
        # labels: (256,)
        # 训练步骤...
        if i >= steps_per_epoch:
            break  # 因为是infinite=True，需要手动break

    # 验证
    for i, (features, labels) in enumerate(val_loader):
        # 验证步骤...
        if i >= val_steps_per_epoch:
            break
```

**工作原理**：
1. **数据划分**（初始化时完成，使用固定种子42）：
   - 从每个dataset中**随机**抽取`val_ratio`比例的样本作为验证集
   - 例如：BTCUSDT_2024有526,921个样本，随机抽取20%=105,384个样本作为验证集，其余421,537个作为训练集
   - 验证集样本在整个时间序列中随机分布，不是连续的一段
   - 训练和验证DataLoader使用相同的划分（通过固定种子确保）
2. **分段采样**：每个batch由`segments_per_batch`个segments组成
3. **智能读取**：每个segment从随机dataset的有效索引中连续取`segment_size`个样本
   - 如果采样的索引是连续的，使用HDF5切片读取（快速）
   - 如果索引不连续，使用列表索引读取
4. 既能实现真正的随机采样（跨标的、跨时间），又尽可能利用HDF5的高效读取

**参数说明**：
- `dataset_names`: None表示使用所有datasets，也可以指定子集
- `val_ratio`: 验证集比例，训练和验证必须使用相同的值
- `batch_size`: 必须能被`segments_per_batch`整除
- `segments_per_batch`: 增大可以提高多样性，减小可以提高I/O效率
- `num_workers`: 建议设置为2-4，每个worker独立打开HDF5文件

### 5. 简单的全内存加载DataLoader

如果数据能完全放入内存，也可以使用简单的方式：

```python
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader

# 一次性加载所有数据
with h5py.File('./data/all_data.h5', 'r') as f:
    datasets = ['BTCUSDT_2022', 'BTCUSDT_2023']
    data_list = [f[name][:] for name in datasets]

data = np.concatenate(data_list, axis=0)
features = torch.from_numpy(data[:, :, :-1])  # (N, 120, 10)
labels = torch.from_numpy(data[:, 0, -1]).long()  # (N,)

# 创建TensorDataset
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)

for features, labels in dataloader:
    print(features.shape, labels.shape)
    break
```

## 查看文件信息

```python
import h5py

with h5py.File('./data/all_data.h5', 'r') as f:
    print("Available datasets:")
    for name in f.keys():
        dataset = f[name]
        print(f"  {name}: shape={dataset.shape}, size={dataset.nbytes/1024/1024:.2f}MB")

    print("\nMetadata:")
    for key, value in f.attrs.items():
        print(f"  {key}: {value}")
```

## 注意事项

1. **Label位置**：Label只存储在`[:, 0, -1]`位置，其他时间步的最后一列是NaN
2. **数据类型**：所有数据都是float32，label需要转换为int进行训练
3. **内存管理**：
   - 小数据集（<8GB）：直接全部读入内存，训练速度最快
   - 大数据集（>8GB）：使用分批读取或PyTorch的lazy loading
4. **Shuffle策略**：
   - 内存充足：读入内存后使用numpy/torch的shuffle
   - 内存不足：预先生成shuffled indices，然后按序读取chunks
