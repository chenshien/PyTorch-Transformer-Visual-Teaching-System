# PyTorch与Transformer可视化教学系统

这是一个用于学习和可视化PyTorch和Transformer模型的教学系统，提供了多种交互式演示和可视化功能，帮助用户深入理解Transformer架构和自注意力机制的工作原理。

## 📋 功能特点

- 🔍 **Transformer架构的可视化展示**：直观展示Transformer的编码器、解码器结构
- 🧠 **自注意力机制（Self-Attention）的交互式演示**：动态呈现注意力权重计算过程
- 📈 **神经网络训练过程可视化**：实时展示模型训练中参数变化和损失函数优化过程
- 🧩 **Transformer各组件工作原理演示**：包括位置编码、多头注意力、前馈网络等
- 🚀 **实际应用案例展示**：文本生成、情感分析、机器翻译等应用示例

## 💻 系统要求

- Python 3.8+
- PyTorch 2.0+
- 建议使用支持中文显示的操作系统环境

## 🔧 安装方法

1. 克隆或下载本项目
2. 安装所需依赖：

```bash
pip install -r requirements.txt
```

3. 确保系统中已安装中文字体（详见下方中文显示问题解决方案）

## 🚀 运行方法

推荐使用启动脚本运行应用程序：

```bash
python start.py
```

启动脚本会自动检查依赖、中文字体支持，并启动应用。

或者直接使用Streamlit运行：

```bash
streamlit run app.py
```

启动后将自动打开浏览器窗口，显示教学系统界面。

## 📚 系统模块说明

- `app.py`: 主应用程序入口，包含界面布局和交互逻辑
- `start.py`: 应用程序启动脚本，处理依赖检查和环境配置
- `models/`: 包含各种模型实现
  - `simple_transformer.py`: 简化版Transformer模型实现
- `visualizations/`: 可视化相关功能
  - `attention_visualization.py`: 注意力机制可视化
  - `transformer_architecture.py`: Transformer架构可视化
  - `training_visualization.py`: 训练过程可视化
- `utils/`: 工具函数
  - `fix_chinese.py`: 中文显示支持
  - `demo_data.py`: 演示数据生成
- `examples/`: 示例数据和使用案例

## 🈶 中文显示问题解决方案

如果遇到中文显示不正常的情况，请尝试以下解决方案：

### Windows系统
Windows系统通常已经内置了中文字体支持，如果显示异常，请确保安装了以下任一中文字体：
- 微软雅黑 (Microsoft YaHei)
- 宋体 (SimSun)
- 黑体 (SimHei)

### macOS系统
可以安装以下字体：
```bash
brew tap homebrew/cask-fonts
brew install --cask font-noto-sans-cjk
```
或下载安装思源黑体 (Source Han Sans)

### Linux系统
对于不同的发行版，请安装相应的中文字体包：

- Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
```

- Fedora:
```bash
sudo dnf install wqy-microhei-fonts wqy-zenhei-fonts
```

- Arch Linux:
```bash
sudo pacman -S wqy-microhei wqy-zenhei
```

使用启动脚本`python start.py`运行应用程序时，系统会自动尝试检测和安装所需的中文字体。

## 📝 使用建议

- 本系统适合初学者到中级学习者，通过交互式界面深入理解Transformer架构和PyTorch框架
- 推荐按照"Transformer架构" → "自注意力机制" → "模型训练可视化" → "实际应用案例"的顺序学习
- 每个模块都提供了交互式调节参数的功能，建议通过调节参数观察变化来加深理解

## 📜 许可证

本项目采用MIT许可证。详见LICENSE文件。 