# Distil-Whisper-Small-ZH: 基于知识蒸馏的中文语音识别模型

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-yellow.svg)](https://github.com/huggingface/transformers)

[English](#english) | [中文](#中文)

</div>

---

## English

### 📋 Project Overview

This project implements **knowledge distillation** for the Whisper speech recognition model, specifically optimized for **Chinese speech recognition tasks**. Through teacher-student distillation, we successfully compressed the Whisper-small model while maintaining high performance, achieving:

- 🎯 **39.09% reduction** in model parameters (from 230.54M to 140.41M)
- ⚡ **2.89x speedup** in inference time
- 📈 **3.16% improvement** in Character Error Rate (CER)

### 🎯 Key Features

- **Model Compression**: Utilizing knowledge distillation to reduce model size without significant performance loss
- **Multi-dataset Training**: Trained on Common Voice, FLEURS, and AISHELL datasets for robust Chinese speech recognition
- **Adaptive Fine-tuning**: Employing AdaLoRA for efficient parameter-efficient fine-tuning
- **Simplified Chinese Output**: Direct output of Simplified Chinese without requiring Traditional-to-Simplified conversion
- **Punctuation Recognition**: Enhanced ability to recognize punctuation marks in speech

### 🗂️ Datasets

#### Common Voice Dataset
An open-source project initiated by Mozilla to collect large-scale speech data across multiple languages.
- **Multi-language Support**: Over 70 languages including Chinese
- **Diversity**: Recordings from speakers of various backgrounds, ages, and genders
- **Open Source**: Freely available for research and development

#### FLEURS Dataset
A multilingual speech command dataset released by Meta AI, supporting cross-language speech recognition research.
- **Multi-language Support**: 102 languages
- **Standardized Commands**: Common voice commands translated across languages
- **High-quality Recordings**: Clear audio with minimal background noise

#### AISHELL Dataset
An open Chinese speech recognition dataset widely used in academic research and industrial applications.
- **Domain**: Chinese Mandarin speech
- **Quality**: Professional recordings with detailed annotations
- **Enhancement**: Punctuation added using Alibaba's punc_ct-transformer

### 🏗️ Model Architecture

#### Knowledge Distillation (Teacher-Student)

The distillation process follows a Teacher-Student paradigm:

- **Teacher Model**: Whisper-small (12 encoder layers + 12 decoder layers)
- **Student Model**: Distil-Whisper-small (12 encoder layers + 2 decoder layers)
- **Initialization Strategy**: Maximum-margin layer copying from teacher to student
- **Training Strategy**: Freeze encoder, train only decoder to optimize efficiency

#### Loss Functions

- **Cross-Entropy Loss**: Measures prediction accuracy against ground truth
- **KL Divergence Loss**: Captures knowledge transfer from teacher to student
- **Combined Loss**: Weighted combination (0.8 × CE + KL weight × KL)

### 🚀 Quick Start

#### Environment Setup

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Configure Accelerate**
   ```bash
   accelerate config
   ```

3. **Hugging Face Authentication**
   ```bash
   git config --global credential.helper store
   huggingface-cli login
   ```

#### Pseudo-labeling

Generate pseudo-labels using the teacher model:

```bash
bash run_pseudo_labelling.sh
```

**Note**: Modify `model_name_or_path`, `dataset_name`, and other parameters in the script before running.

#### Model Initialization

Create and initialize the student model:

```bash
python create_student_model.py \
  --teacher_checkpoint "openai/whisper-small" \
  --encoder_layers 12 \
  --decoder_layers 2 \
  --save_dir "./distil-small-init"
```

The initialized student model will be saved to `./distil-small-init`.

#### Distillation Training

Execute distillation training script:

```bash
bash run_distillation.sh
```

The trained model will be saved to `./model`. In this project, the encoder is frozen and only the decoder is trained (configured via `freeze_encoder` parameter).

#### Model Evaluation

Evaluate the model performance:

```bash
bash run_eval_sf.sh
```

Evaluation metrics include:
- **CER** (Character Error Rate): Measures character-level recognition accuracy
- **RTF** (Real-Time Factor): Ratio of audio duration to processing time

### 📊 Experimental Results

#### Performance Comparison

| Model | Parameters | CER (avg) | Speed-up | Dataset |
|-------|-----------|-----------|----------|---------|
| Whisper-small | 230.54M | 17.43% | 1.0x | Common Voice + FLEURS + AISHELL |
| Distil-Whisper (before fine-tuning) | 140.41M | 27.53% | 2.26x | Common Voice + FLEURS + AISHELL |
| **Distil-Whisper-finetune** | **140.41M** | **17.11%** | **2.89x** | **Common Voice + FLEURS + AISHELL** |

#### Detailed Results by Dataset

**Whisper-small Performance:**

| Dataset | Size | CER | Validation Time (s) |
|---------|------|-----|---------------------|
| Common Voice | 10,626 | 21.51% | 9,767 |
| FLEURS | 945 | 16.87% | 1,718 |
| AISHELL | 7,176 | 13.92% | 4,990 |
| **Average** | - | **17.43%** | - |

**Distil-Whisper-finetune Performance:**

| Dataset | Size | CER | CER Change | Validation Time (s) | Time Reduction | Speed-up |
|---------|------|-----|------------|---------------------|----------------|----------|
| Common Voice | 10,626 | 18.73% | ↓12.93% | 2,511 | 74.29% | 2.89x |
| FLEURS | 945 | 24.54% | ↑45.51% | 459 | 73.28% | 2.74x |
| AISHELL | 7,176 | 8.06% | ↓42.06% | 1,240 | 76.53% | 3.02x |
| **Average** | - | **17.11%** | **↓3.16%** | - | **74.70%** | **2.89x** |

*Validation time measured on 2×NVIDIA T4 GPUs with batch_size=64*

#### Key Findings

1. **Model Compression**: Successfully reduced model size by 39.09% (90M parameters)
2. **Inference Speed**: Achieved 2.89x speedup in inference time
3. **Recognition Accuracy**: Improved CER by 3.16% on average compared to the teacher model
4. **Language Adaptation**: Fine-tuned model shows superior performance on Chinese datasets compared to the original multilingual model
5. **Punctuation Recognition**: Enhanced ability to recognize and output punctuation marks
6. **Direct Simplified Output**: Eliminates the need for Traditional-to-Simplified Chinese conversion

### 📈 Training Process

#### Loss Curves

**Training Loss Progression:**
- **Initial Phase** (0-1000 steps): Rapid loss decrease as the model learns basic patterns
- **Middle Phase** (1000-2000 steps): Gradual loss reduction with slower convergence
- **Convergence Phase** (2000+ steps): Loss stabilizes around 0.2, indicating optimal training state

**Fine-tuning Loss:**
- Initial rapid decrease from ~0.55 to ~0.35
- Steady decline to ~0.2 over 10,000 steps
- Final convergence demonstrates effective adaptation to Chinese datasets

### 🛠️ Additional Tools

- **`test_whisper.py`**: Perform inference on online datasets
- **`test_whisper_local.py`**: Perform inference on local WAV audio files
- **`count_params.py`**: Calculate model parameter count

### 📁 Project Structure

```
Distil-whisper-small-zh/
├── cer/                          # Character Error Rate evaluation
│   ├── cer.py                    # CER metric implementation
│   └── data_utils.py             # Data utilities for CER calculation
├── create_student_model.py       # Student model initialization
├── run_distillation.py           # Main distillation training script
├── run_distillation.sh           # Distillation training shell script
├── run_pseudo_labelling.py       # Pseudo-label generation
├── run_pseudo_labelling.sh       # Pseudo-labeling shell script
├── run_eval.py                   # Model evaluation script
├── run_eval_sf.sh                # Evaluation shell script
├── test_whisper.py               # Online dataset inference
├── test_whisper_local.py         # Local audio file inference
├── count_params.py               # Parameter counting utility
├── data_utils.py                 # Data processing utilities
├── setup.py                      # Package setup configuration
└── README.md                     # Project documentation
```

### 🔬 Methodology

#### Three-Phase Experimental Design

**Phase 1: Distillation**
- Extract pseudo-labels from teacher model
- Initialize student model with maximum-margin layer copying
- Train with combined CE and KL divergence loss
- Freeze encoder, train only decoder

**Phase 2: Fine-tuning**
- Apply AdaLoRA for parameter-efficient fine-tuning
- Train on combined Common Voice, FLEURS, and AISHELL datasets
- Enhance Chinese language adaptation
- Improve punctuation recognition

**Phase 3: Evaluation**
- Test on held-out datasets
- Compare with baseline models (Whisper-base, Whisper-tiny)
- Measure CER and RTF metrics
- Conduct qualitative analysis on real recordings

### 📖 References

1. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. arXiv preprint arXiv:2212.04356.
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
3. Hugging Face. (2021). Distil-Whisper: Distilling OpenAI's Whisper for Faster, Smaller Models.

---

## 中文

### 📋 项目简介

本项目针对 Whisper 语音识别模型实现了**知识蒸馏**技术，专门优化用于**中文语音识别任务**。通过教师-学生蒸馏方法，我们成功压缩了 Whisper-small 模型，同时保持了高性能表现，实现了：

- 🎯 模型参数**减少 39.09%**（从 230.54M 降至 140.41M）
- ⚡ 推理速度**提升 2.89 倍**
- 📈 字符错误率（CER）**降低 3.16%**

### 🎯 核心特性

- **模型压缩**：利用知识蒸馏技术在不显著损失性能的前提下减小模型规模
- **多数据集训练**：在 Common Voice、FLEURS 和 AISHELL 数据集上训练，实现鲁棒的中文语音识别
- **自适应微调**：采用 AdaLoRA 进行高效的参数微调
- **简体中文输出**：直接输出简体中文，无需繁简转换
- **标点符号识别**：增强了对语音中标点符号的识别能力

### 🗂️ 数据集介绍

#### Common Voice 数据集
Mozilla 发起的开源项目，旨在收集多种语言的大规模语音数据集。
- **多语言支持**：支持超过 70 种语言，包括中文
- **多样性**：来自不同背景、年龄和性别的说话者录音
- **开源性**：免费提供用于研究和开发

#### FLEURS 数据集
Meta AI 发布的多语言语音命令数据集，支持跨语言语音识别研究。
- **多语言支持**：支持 102 种语言
- **标准化命令**：跨语言翻译的标准语音命令
- **高质量录音**：清晰的音频，背景噪音少

#### AISHELL 数据集
开放的中文语音识别数据集，广泛应用于学术研究和工业应用。
- **领域**：中文普通话语音
- **质量**：专业录音，带有详细标注
- **增强处理**：使用阿里达摩院的 punc_ct-transformer 添加标点符号

### 🏗️ 模型架构

#### 知识蒸馏（Teacher-Student 模式）

蒸馏过程遵循教师-学生范式：

- **教师模型**：Whisper-small（12 层编码器 + 12 层解码器）
- **学生模型**：Distil-Whisper-small（12 层编码器 + 2 层解码器）
- **初始化策略**：从教师模型最大间隔层复制权重到学生模型
- **训练策略**：冻结编码器，仅训练解码器以优化效率

#### 损失函数

- **交叉熵损失**：衡量预测准确度与真实标签的差异
- **KL 散度损失**：捕获从教师到学生的知识转移
- **组合损失**：加权组合（0.8 × 交叉熵 + KL 权重 × KL 散度）

### 🚀 快速开始

#### 环境配置

1. **安装依赖**
   ```bash
   pip install -e .
   ```

2. **配置 Accelerate**
   ```bash
   accelerate config
   ```

3. **Hugging Face 身份验证**
   ```bash
   git config --global credential.helper store
   huggingface-cli login
   ```

#### 伪标签提取

使用教师模型生成伪标签：

```bash
bash run_pseudo_labelling.sh
```

**注意**：运行前请修改脚本中的 `model_name_or_path`、`dataset_name` 等参数。

#### 模型初始化

创建并初始化学生模型：

```bash
python create_student_model.py \
  --teacher_checkpoint "openai/whisper-small" \
  --encoder_layers 12 \
  --decoder_layers 2 \
  --save_dir "./distil-small-init"
```

初始化的学生模型将保存到 `./distil-small-init`。

#### 蒸馏训练

执行蒸馏训练脚本：

```bash
bash run_distillation.sh
```

训练完成的模型将保存到 `./model`。在本项目中，编码器被冻结，仅训练解码器（通过 `freeze_encoder` 参数配置）。

#### 模型评估

评估模型性能：

```bash
bash run_eval_sf.sh
```

评估指标包括：
- **CER**（字符错误率）：衡量字符级别的识别准确度
- **RTF**（反实时因子）：音频时长与处理时间的比值

### 📊 实验结果

#### 性能对比

| 模型 | 参数量 | CER（平均） | 速度提升 | 数据集 |
|------|--------|------------|---------|--------|
| Whisper-small | 230.54M | 17.43% | 1.0x | Common Voice + FLEURS + AISHELL |
| Distil-Whisper（微调前） | 140.41M | 27.53% | 2.26x | Common Voice + FLEURS + AISHELL |
| **Distil-Whisper-finetune** | **140.41M** | **17.11%** | **2.89x** | **Common Voice + FLEURS + AISHELL** |

#### 各数据集详细结果

**Whisper-small 表现：**

| 数据集 | 大小 | CER | 验证耗时（秒） |
|--------|------|-----|---------------|
| Common Voice | 10,626 | 21.51% | 9,767 |
| FLEURS | 945 | 16.87% | 1,718 |
| AISHELL | 7,176 | 13.92% | 4,990 |
| **平均** | - | **17.43%** | - |

**Distil-Whisper-finetune 表现：**

| 数据集 | 大小 | CER | CER 变化 | 验证耗时（秒） | 耗时下降 | 速度提升 |
|--------|------|-----|---------|---------------|---------|---------|
| Common Voice | 10,626 | 18.73% | ↓12.93% | 2,511 | 74.29% | 2.89x |
| FLEURS | 945 | 24.54% | ↑45.51% | 459 | 73.28% | 2.74x |
| AISHELL | 7,176 | 8.06% | ↓42.06% | 1,240 | 76.53% | 3.02x |
| **平均** | - | **17.11%** | **↓3.16%** | - | **74.70%** | **2.89x** |

*验证耗时在 2×NVIDIA T4 GPU 上测量，batch_size=64*

#### 主要发现

1. **模型压缩**：成功将模型规模减小 39.09%（减少 9000 万参数）
2. **推理速度**：推理时间提升 2.89 倍
3. **识别准确度**：相比教师模型，CER 平均降低 3.16%
4. **语言适应**：微调后的模型在中文数据集上表现优于原始多语言模型
5. **标点符号识别**：增强了对标点符号的识别和输出能力
6. **简体中文直接输出**：无需进行繁简转换

### 📈 训练过程

#### 损失曲线

**训练损失变化：**
- **初始阶段**（0-1000 步）：损失快速下降，模型学习基础模式
- **中间阶段**（1000-2000 步）：损失逐渐减小，收敛速度放缓
- **收敛阶段**（2000+ 步）：损失稳定在 0.2 左右，达到最优训练状态

**微调损失：**
- 初始快速下降，从约 0.55 降至约 0.35
- 在 10,000 步内稳步下降至约 0.2
- 最终收敛表明成功适应中文数据集

### 🛠️ 其他工具

- **`test_whisper.py`**：在在线数据集上进行推理
- **`test_whisper_local.py`**：在本地 WAV 音频文件上进行推理
- **`count_params.py`**：计算模型参数量

### 📁 项目结构

```
Distil-whisper-small-zh/
├── cer/                          # 字符错误率评估模块
│   ├── cer.py                    # CER 指标实现
│   └── data_utils.py             # CER 计算数据工具
├── create_student_model.py       # 学生模型初始化
├── run_distillation.py           # 主蒸馏训练脚本
├── run_distillation.sh           # 蒸馏训练 Shell 脚本
├── run_pseudo_labelling.py       # 伪标签生成
├── run_pseudo_labelling.sh       # 伪标签 Shell 脚本
├── run_eval.py                   # 模型评估脚本
├── run_eval_sf.sh                # 评估 Shell 脚本
├── test_whisper.py               # 在线数据集推理
├── test_whisper_local.py         # 本地音频文件推理
├── count_params.py               # 参数计数工具
├── data_utils.py                 # 数据处理工具
├── setup.py                      # 包配置文件
└── README.md                     # 项目文档
```

### 🔬 研究方法

#### 三阶段实验设计

**阶段一：蒸馏**
- 从教师模型提取伪标签
- 通过最大间隔层复制初始化学生模型
- 使用交叉熵和 KL 散度组合损失训练
- 冻结编码器，仅训练解码器

**阶段二：微调**
- 应用 AdaLoRA 进行参数高效微调
- 在 Common Voice、FLEURS 和 AISHELL 组合数据集上训练
- 增强中文语言适应能力
- 改善标点符号识别

**阶段三：评估**
- 在独立测试集上测试
- 与基线模型（Whisper-base、Whisper-tiny）对比
- 测量 CER 和 RTF 指标
- 对真实录音进行定性分析

### 🎓 研究背景与相关工作

#### Whisper 模型介绍
Whisper 是 OpenAI 开发的先进语音识别模型，在英文任务上表现优异，具有出色的速度和相对较小的模型体积。本项目基于 Whisper-small 模型，通过知识蒸馏技术实现模型压缩和优化。

#### 知识蒸馏在 NLP 领域的应用
知识蒸馏是一种将大型模型知识转移给小型模型的技术，在自然语言处理领域已展现出显著潜力。通过将教师模型的知识蒸馏到学生模型，可以在性能几乎不受影响的情况下显著减少模型大小和计算需求。

#### 与其他模型的对比

本项目选择 Whisper-small 作为教师模型的原因：
- **Whisper-small**：12 层编码器 + 12 层解码器
- **Whisper-base**：6 层编码器 + 6 层解码器  
- **Whisper-tiny**：4 层编码器 + 4 层解码器

Whisper-base 和 Whisper-tiny 的模型结构相对简单，知识蒸馏对其压缩效果有限，因此选择结构更复杂的 Whisper-small 进行蒸馏。

### 📖 参考文献

1. Radford, A., et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision. arXiv preprint arXiv:2212.04356.
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
3. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
4. Hugging Face. (2021). Distil-Whisper: Distilling OpenAI's Whisper for Faster, Smaller Models.

### 🙏 致谢

感谢 OpenAI 提供 Whisper 模型，感谢 Hugging Face 提供 Distil-Whisper 框架和工具支持。

---

<div align="center">

**如果本项目对您有帮助，欢迎 Star ⭐**

Made with ❤️ by Team 12

</div>
