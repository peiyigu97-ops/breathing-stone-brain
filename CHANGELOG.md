# Changelog

---

## [2.0.0] — 2026-07-24

### HumanBrainDT: Major Brain Model Upgrade

#### 新增脑区 (10 → 16 regions)

新增 6 个解剖学关键区域，覆盖此前模型中缺失的核心通路：

| 新增区域 | 功能角色 |
|----------|----------|
| **Insula（脑岛）** | 内感受核心、痛觉情感层、恶心/肌肉张力/自主神经驱动 |
| **Anterior Cingulate Cortex（前扣带回）** | 冲突监测、痛情感处理、焦虑调控 |
| **Orbitofrontal Cortex（眶额皮质）** | 奖励估值、冲动抑制、嗅觉/味觉整合 |
| **Hypothalamus（下丘脑）** | HPA 轴起搏器、体温调节、疲劳信号、自主神经总驱动 |
| **Locus Coeruleus（蓝斑核）** | 去甲肾上腺素广播、应激唤醒放大器、全脑警觉调制 |
| **Substantia Nigra（黑质）** | 多巴胺通路、运动启动、奖励学习 |

#### 新增感觉通道

```
nausea_input      — 恶心/消化道信号 → 脑干(NTS) → 脑岛 → 下丘脑
muscle_tension    — 骨骼肌张力 → 脑岛 → 小脑 → 顶叶
fatigue_input     — 疲劳/耗竭 → 下丘脑 → 蓝斑核 → 脑干
stress_hormone    — 皮质醇/CRH → 下丘脑 → 蓝斑核 → 杏仁核
dopamine_signal   — 相位性多巴胺 → 黑质 → 基底核 → 眶额皮质
olfactory_input   — 嗅觉 → 眶额皮质 → 杏仁核 → 海马
taste_input       — 味觉 → 眶额皮质 → 脑岛
conflict_input    — 认知冲突 → 前扣带回 → 额叶
```

#### 神经通路修正 (CHANNEL_ROUTING 全面重写)

所有通路现在走解剖学正确的目标区域，例如：
- `pain_input` → 脑干 + **脑岛** + 杏仁核 + **前扣带回** + 丘脑
- `interoception` → **脑岛** → 下丘脑 → 杏仁核（不再用顶叶代理）
- `threat_input` → 杏仁核 + 脑干 + **蓝斑核**
- `stress_hormone` → 下丘脑 → 蓝斑核 → 杏仁核

#### ANS 计算升级

- 蓝斑核(LC) NE 广播现在显式放大交感神经张力
- 脑岛活动现在贡献副交感神经调节
- 皮质醇通道直接驱动 HPA 轴计算

#### 焦虑分解升级

- `somatic` 焦虑现在从脑岛 + 蓝斑核计算（不再从顶叶代理）
- `anticipatory` 加入 ACC 冲突信号
- `regulation` 加入 ACC 辅助 PFC 调控

#### KMAP 扩展 (40 → 120+ 关键词)

新增覆盖：`心跳加速`、`肌肉紧绷`、`瞳孔扩张`、`恶心`、`精疲力竭`、`崩溃`、`快乐`、`多巴胺`、`冲突`、`犹豫`等日常生理/情绪词汇（中英双语）。

#### 传导延迟扩充

新增 40+ 条传导延迟记录（ms），覆盖所有新通路（如 `fatigue_input → locus_coeruleus: 10ms`，`stress_hormone → amygdala: 20ms`）。

---

### HumanBrainDT: 3D 可视化 Galaxy View 重构

#### 渲染方式从粒子云改为 Galaxy 星云风格

旧版：每个脑区是解剖形状粒子云（sheet/ellipsoid/curved）  
新版：每个脑区 = **主星 + 星云 + 卫星小星** 三层结构

| 层级 | 视觉元素 | 说明 |
|------|----------|------|
| Primary Star | 白热亮点，size 0.08–0.18 | 脑区核心，始终可见，闪烁 |
| Nebula Cloud | 小圆形精灵粒子，size 0.014–0.020 | 填充解剖体积，静止可见 |
| Satellites | 中等亮度小星，size 0.025–0.055 | 散布于云团中段 |

所有粒子使用 128px 圆形 canvas 纹理，消除方形像素感。

#### 分层激活时序

激活时三层按序点亮，有明确时间差：
1. **神经连线**（filament）— 即时响应（lerp 0.60）
2. **主星** — 快速激活（lerp 0.10）
3. **星云** — 延迟跟随（lerp 0.028，约滞后 1.5s）

静止状态三层均可见，主星始终亮于星云。

#### 脑区外层星云（三层）

```
Layer 1: 内填充 (r=0.5–1.35)  — 3500 粒子，填充脑区间隙
Layer 2: 中层壳 (r=1.35–2.0)  — 5000 粒子，主要可见星云晕
Layer 3: 外层晕 (r=2.0–3.2)  — 4000 粒子，漫反射边缘光
```

#### MNI 坐标放大 1.6×

所有脑区坐标按 MNI 标准空间缩放 1.6× 增大区域间距，避免视觉重叠，同时保持解剖拓扑关系正确。

#### 神经连线重构

旧版：TubeGeometry（体积管道）  
新版：LineSegments + 顶点颜色缓冲（单次 drawcall），激活时蓝白闪亮，明显亮于星云。

#### 脑区颜色区分

16 个脑区分配视觉上截然不同的颜色，静止即可区分：

| 脑区 | 颜色 |
|------|------|
| Frontal Cortex | 钢蓝 `#4a90d9` |
| Parietal Cortex | 天蓝 `#7ec8e3` |
| Temporal Cortex | 海绿 `#48d1a0` |
| Occipital Cortex | 青蓝 `#38c0c8` |
| Insula | 紫罗兰 `#b060d8` |
| Anterior Cingulate | 玫红 `#e070a0` |
| Orbitofrontal | 琥珀 `#f0a040` |
| Basal Ganglia | 红 `#d84848` |
| Thalamus | 黄绿 `#a0c060` |
| Hippocampus | 天蓝 `#40a8e0` |
| Amygdala | 深红 `#e04060` |
| Hypothalamus | 橙 `#f08020` |
| Locus Coeruleus | 青 `#30d8f0` |
| Substantia Nigra | 金 `#e8c020` |
| Cerebellum | 绿 `#50d070` |
| Brain Stem | 灰蓝 `#a0b8d0` |

#### 布局与渲染质量

- 视口自适应（`window.innerWidth / innerHeight`），不再锁定固定分辨率
- 渲染分辨率 devicePixelRatio × 2，canvas 纹理 128px，消除模糊
- 面板改为半透明（opacity 0.42），不遮挡脑区视图
- 右侧面板收窄至 480px
- 滚轮缩放由 OrbitControls 原生处理
- 旋转场景背景（3000 星点旋转星场）

---

### 过程归档文件

| 文件 | 说明 |
|------|------|
| `index_v1_particle_3d.html` | v1：解剖形状粒子云版本 |
| `index_v2_galaxy_cluster.html` | v2：球形粒子集群版本 |
| `index.html` | 当前：Galaxy 主星 + 星云版本 |

---

## [1.0.0] — 2026-03-21

初始版本。CHIMERA Drosophila 连接组仿真 + HumanBrainDT 10 区域 LIF 模型 + Breathing Stone 桥接。详见 `HANDOFF.md`。
