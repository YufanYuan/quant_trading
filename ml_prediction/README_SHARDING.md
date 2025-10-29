# 大数据集分片加载方案

## 问题背景

原始数据集 `train_features.npy` 和 `train_labels.npy` 总大小约 **42GB**，一次性加载到内存会导致内存不足。

## 解决方案

### 核心策略

1. **数据类型优化**：float64 → float32, int64 → int32（节省50%空间）
2. **预先Shuffle**：在分片前对索引进行shuffle（不需要加载全部数据到内存）
3. **数据分片**：将大文件切分成多个小文件
4. **灵活加载**：训练时可选择一次性加载或逐片加载

## 文件说明

### 1. `data_loader.py`（已修改）
- 添加了数据类型转换：自动将features转为float32，labels转为int32
- 生成新数据时会自动使用优化的数据类型

### 2. `split_dataset.py`（新增）
数据集分片工具，支持：
- **预先shuffle**：只shuffle索引数组（约36MB），然后按shuffled索引读取数据
- **数据类型转换**：可选将float64转为float32
- **分片处理**：将大文件切分成多个小shard

**关键参数**：
- `samples_per_shard`：每个shard包含的样本数（默认10000）
- `shuffle`：是否预先shuffle数据（默认True）
- `random_seed`：随机种子（默认42）

### 3. `sharded_dataset.py`（新增）
PyTorch Dataset实现，支持：
- **ShardDataset**：单个shard的数据集（直接加载到内存）
- **MultiShardDataset**：多shard管理器
  - `get_all_shards_dataset()`：一次性加载所有shard（需要足够内存）
  - `get_single_shard_dataset()`：加载单个shard
  - `iter_shards()`：迭代shard（可选shuffle顺序）

### 4. `train_sharded.py`（新增）
使用分片数据集的训练脚本，基于原有的 `train.py`，支持：
- 加载分片数据集
- 自动划分训练集/验证集
- 完整的训练流程

## 使用流程

### Step 1: 分片现有数据集

```bash
# 方式1：转换数据类型 + 分片
uv run python ml_prediction/split_dataset.py
# 选择选项1，会：
# 1. 创建 data/shards_float32/train_features_float32.npy（约21GB）
# 2. 创建 data/shards_float32/train_labels_int32.npy
# 3. 分片成多个 features_shard_XXX.npy 和 labels_shard_XXX.npy
# 4. 保存元数据 metadata.npy

# 方式2：直接分片原始数据（不转换类型）
# 选择选项2
```

**预期输出**：
```
data/shards_float32/
├── metadata.npy                    # 元数据
├── train_features_float32.npy     # 转换后的完整文件（可选择删除以节省空间）
├── train_labels_int32.npy
├── features_shard_000.npy         # 分片文件
├── labels_shard_000.npy
├── features_shard_001.npy
├── labels_shard_001.npy
├── ...
└── features_shard_XXX.npy
```

### Step 2: 使用分片数据集训练

```bash
uv run python ml_prediction/train_sharded.py
```

**训练脚本配置**：
```python
config = {
    'data_mode': 'load_all',  # 一次性加载所有shard
    'data_dir': './data/shards_float32',  # 分片数据目录
    'batch_size': 64,
    'num_epochs': 100,
    ...
}
```

## 内存优化对比

### 原始方案
- Features: 42.32 GB (float64)
- Labels: 0.04 GB (int64)
- **总计: 约 42.36 GB**

### 优化后
- Features: 21.16 GB (float32)
- Labels: 0.02 GB (int32)
- **总计: 约 21.18 GB（节省50%）**

### 分片后
- 每个shard约 **45 MB**（samples_per_shard=10000）
- 训练时可以：
  - 一次加载所有shard：21GB内存
  - 一次加载单个shard：45MB内存（配合shard轮转训练）

## Shuffle策略

### 为什么预先shuffle可行？

1. **只shuffle索引**：索引数组很小（4.7M样本 × 8字节 ≈ 36MB）
2. **按索引读取**：使用memmap从原始文件按shuffled索引读取数据
3. **分批处理**：每次读取10000个样本，内存占用可控

### Shuffle效果

数据已在分片时完全随机化，训练时：
- **简单方案**：顺序读取各shard，DataLoader内部shuffle
- **进阶方案**：每个epoch随机选择shard顺序 + DataLoader shuffle

两种方案都能保证数据的随机性。

## 进阶用法

### 自定义分片参数

编辑 `split_dataset.py` 的 `main()` 函数：

```python
split_dataset(
    features_path,
    labels_path,
    output_dir,
    samples_per_shard=10000,  # 修改每个shard的样本数
    shuffle=True,              # 是否预先shuffle
    random_seed=42             # 随机种子
)
```

### 测试数据加载

```bash
# 测试分片数据集加载
uv run python ml_prediction/sharded_dataset.py
```

## 注意事项

1. **磁盘空间**：分片会创建新文件，确保有足够磁盘空间（至少50GB）
2. **删除中间文件**：分片完成后，可以删除 `train_features_float32.npy` 节省空间
3. **原始文件保留**：建议保留原始的 `train_features.npy` 作为备份
4. **Shard大小选择**：
   - 太小：文件太多，管理麻烦
   - 太大：单个shard占用内存多
   - 推荐：10000-100000样本/shard

## 未来改进

- [ ] 实现真正的"逐shard训练"模式（当前需要加载所有shard）
- [ ] 添加数据增强支持
- [ ] 支持分布式训练
- [ ] 优化跨shard读取性能