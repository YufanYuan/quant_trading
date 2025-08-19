# 1 分钟剥头皮策略 – 实现规范 V1.1

（面向代码实现的技术说明）

## 0. 术语

| 术语             | 定义                                                                                                                                                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **原始 K 线**    | 来自交易所的 1 分钟 OHLCV（Open/High/Low/Close/Volume）。                                                                                               |
| **HA-K 线**      | 由原始 K 线实时计算得到的 Heikin Ashi OHLC。公式见 §2.2。                                                                                               |
| **干净反色蜡烛** | 与当前趋势颜色相反且无影线（主趋势为上升趋势时，反色蜡烛是收跌 K 线，且无上影线；主趋势为下降趋势时，反色蜡烛是收涨 K 线，且无下影线）的 HA-K 线。      |
| **Doji**         | HA-K 线满足：<br>• 实体高度占全场 ≤ 10 % （也就是说 \| Close-Open \| / (High-Low) <= 10% ）<br>• 上影线 / 下影线长度之比 ∈ \[0.8, 1.25\] (二者近似相等) |

---

## 1. 数据接口

### 1.1 回调函数

```python
Strategy.on_kline(
    ts: datetime,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float
) -> None
```

- **数据节奏**：调用频率 = 原始 1 分钟 K 线节奏；严格按时间顺序。

### 1.2 输出接口（由外部应用实现）

策略调用外部回调以下达/平仓：

```python
broker.open_long(price, sl, tp)   # 做多
broker.open_short(price, sl, tp)  # 做空
broker.close_position(position_id, price)
```

> 说明   策略本身不关心撮合、手续费、滑点；外部 `broker` 层可通过 **fee_handler** 来插入成本计算。

---

## 2. 指标计算

### 2.1 100 EMA（顺势判定基准）

- 计算输入：**HA_Close**（而非原始收盘价）
- EMA 周期：100
- 更新频率：每根新 HA-K 线计算一次

### 2.2 Heikin Ashi 公式

设 $O,H,L,C$ 为**原始** K 线数据，$HA\_O,HA\_H,HA\_L,HA\_C$ 为 HA-K 线：

$$
\begin{aligned}
HA\_C &= \frac{O+H+L+C}{4} \\
HA\_O &= \frac{prev\_HA\_O + prev\_HA\_C}{2} \quad (\text{第一根用 } O) \\
HA\_H &= \max(H, HA\_O, HA\_C)\\
HA\_L &= \min(L, HA\_O, HA\_C)
\end{aligned}
$$

---

## 3. 交易流程

### 3.1 可选交易时段过滤

- **默认**：仅处理 **美国中部时间 (CT) 09 : 00 – 11 : 00** 内的 K 线
- 通过策略构造参数
  `enable_session_filter: bool = True`
  `session_start: time = time(9,0)`
  `session_end: time = time(11,0)`
  可自由关闭或调整。

### 3.2 方向与主趋势判定

| 条件                                   | 主趋势方向   | 主趋势颜色 | 反色颜色 |
| -------------------------------------- | ------------ | ---------- | -------- |
| **HA_Open 和 HA_Close 均高于 100 EMA** | 上升（多单） | 绿色       | 红色     |
| **HA_Open 和 HA_Close 均低于 100 EMA** | 下降（空单） | 红色       | 绿色     |

> 只要 Open 或 Close 任何一个落到 EMA 另一侧，就视为穿越，所有计数归零，重新寻找主趋势。

### 3.3 回调段

- 在 **主趋势同侧**，但 HA-K 线颜色 **反色** 的连续区段。

  - 上升主趋势 → 回调段为红色 (下跌) HA-K
  - 下降主趋势 → 回调段为绿色 (上涨) HA-K

- 回调段长度 ≥ 1 根即可记为一次“回调”。

### 3.4 回调修复判定

回调段之后，若出现 **连续 ≥ 3 根主趋势颜色** 的 HA-K，且这 3 根的 HA_Open 和 HA_Close 全部仍在 EMA 同侧，则回调计数 + 1，视为“一次完整的回调 & 复原”。

### 3.5 开始等待入场时机

当 **回调计数 ≥ 2**，进入 **第 3、4 或更多次 回调段** 时，才开始在该回调段寻找入场机会：

1. **两根连续干净反色蜡烛**（与回调颜色一致、无上/下影线）
2. 当前 HA-K 线为 **高量 Doji**（定义见 §0 术语）
3. 当前 HA-K 线的上下影线长度 **同时大于** 前两根干净反色 HA-K 线各自影线长度（主趋势为下降趋势，干净反色蜡烛为绿色上升蜡烛且仅有上影线；主趋势为上升趋势，干净反色蜡烛为红色下降蜡烛且仅有下影线）

满足后立即触发开仓；若信号失效（下一根开盘未成交），撤单并等待下一轮完整流程。

### 3.5 入场信号 – 高量 Doji

1. 当前 HA-K 线为 **Doji**（见 §0 术语）。
2. 上下影线长度 **同时大于** _前两根_ 干净反色 HA-K 线各自影线长度。
3. 符合 1 & 2 时：

   - **当条 Doji 收盘**立即触发开仓指令
   - 价格、止损、止盈参数见 §4.1
   - 如果下根 K 线开盘时尚未成交 → 撤单，等待下一轮完整流程。

---

## 4. 头寸管理与风控

### 4.1 开仓参数

| 项     | 多单                               | 空单                              |
| ------ | ---------------------------------- | --------------------------------- |
| 入场价 | Doji High                          | Doji Low                          |
| 止损价 | Doji Low 以下                      | Doji High 以上                    |
| 止盈价 | Doji High + (Doji High - Doji Low) | Doji Low - (Doji High - Doji Low) |

> 策略**固定 1:1 RR**；暂不支持动态调整。

### 4.2 持仓规则

- **每标的每方向仅保留一笔头寸**

  - 若已持有多单，再出现新多信号 → 忽略
  - 平仓后方可重新捕捉下一信号

- 止损/止盈触发方式：

  - 实时监控 tick 或次级别价格；一旦触达即通过市价成交（由外层 `broker` 执行）。

### 4.3 手续费与滑点

- 构造函数可注入 `fee_handler: Callable[[Trade], float]`

  - 策略内部调用它以获得交易成本并计入净盈亏
  - 若为 `None` 表示成本忽略，由外部自行处理

---

## 5. 策略类接口草案（供实现时参考）

```python
class Scalp1MinStrategy:
    def __init__(
        self,
        fee_handler: Optional[Callable[[dict], float]] = None,
        enable_session_filter: bool = True,
        session_start: time = time(9, 0),
        session_end: time = time(11, 0),
    ):
        ...

    # === 核心入口 ===
    def on_kline(
        self,
        ts: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ):
        """
        按顺序喂入 1 分钟原始 K 线。
        内部完成：
          1. 生成 HA-K 线
          2. 更新 100EMA
          3. 识别趋势段与回调峰值
          4. 筛选干净反色 & Doji
          5. 触发开仓 / 平仓指令（调用外部 broker）
        """
        ...
```

> **说明**：实现时请保持框架无平台依赖，仅依赖 `datetime`, `math`, `collections`, `pandas`/`numpy` 等通用库。

---

## 6. 后续扩展留口

- 参数化：EMA 周期、Doji 影线比例、RR、趋势判定根数可暴露为可调参数
- 多标的：在外层实例化多个 `Scalp1MinStrategy` 并分别喂数据
- 实盘：外部 Application 负责

  - 数据采集 (WebSocket/API)
  - 调用 `on_kline`
  - 封装 `broker` 与手续费逻辑
  - 状态落盘、风险监控
