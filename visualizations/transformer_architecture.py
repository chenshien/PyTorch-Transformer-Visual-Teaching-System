import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import io
from utils.demo_data import get_demo_data
from utils.fix_chinese import set_chinese_font
import matplotlib.font_manager as fm
import os
import platform
from matplotlib.font_manager import FontProperties

def force_set_chinese_font():
    """强制设置中文字体，尝试多种方法"""
    # 方法1: 设置默认字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 方法2: 检查并直接使用系统字体
    if platform.system() == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',   # 宋体
            'C:/Windows/Fonts/simkai.ttf',   # 楷体
            'C:/Windows/Fonts/simfang.ttf'   # 仿宋
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                # 直接加载字体文件
                zh_font = FontProperties(fname=font_path)
                # 设置为默认字体
                plt.rcParams['font.sans-serif'] = [os.path.basename(font_path).split('.')[0]] + plt.rcParams['font.sans-serif']
                # 全局注册字体
                try:
                    fm.fontManager.addfont(font_path)
                    print(f"已全局注册字体: {font_path}")
                except:
                    pass
                return zh_font
    elif platform.system() == 'Darwin':  # macOS
        # macOS字体路径
        font_paths = [
            "/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            os.path.expanduser("~/Library/Fonts/Noto Sans CJK SC.ttc")
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                zh_font = FontProperties(fname=font_path)
                try:
                    fm.fontManager.addfont(font_path)
                except:
                    pass
                return zh_font
    
    # 方法3: 尝试使用外部字体
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emergency_font_path = os.path.join(script_dir, "utils", "emergency_font.ttf")
    if os.path.exists(emergency_font_path):
        zh_font = FontProperties(fname=emergency_font_path)
        try:
            fm.fontManager.addfont(emergency_font_path)
        except:
            pass
        return zh_font
    
    # 方法4: 尝试下载并使用Google Noto字体
    try:
        # 尝试下载Noto Sans字体作为最后手段
        import urllib.request
        noto_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
        noto_path = os.path.join(script_dir, "utils", "noto_sans_sc.otf")
        if not os.path.exists(noto_path):
            urllib.request.urlretrieve(noto_url, noto_path)
            print(f"已下载Noto Sans SC字体到: {noto_path}")
        
        if os.path.exists(noto_path):
            zh_font = FontProperties(fname=noto_path)
            try:
                fm.fontManager.addfont(noto_path)
            except:
                pass
            return zh_font
    except:
        print("无法下载Google Noto字体")
    
    # 如果所有方法都失败，返回默认字体
    return FontProperties(family='sans-serif')

def apply_chinese_font_to_plot(fig, zh_font):
    """将中文字体应用到图表的所有文本元素"""
    for ax in fig.get_axes():
        # 设置标题字体
        title = ax.get_title()
        if title:
            ax.set_title(title, fontproperties=zh_font)
        
        # 设置x轴和y轴标签字体
        xlabel = ax.get_xlabel()
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=zh_font)
        
        ylabel = ax.get_ylabel()
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=zh_font)
        
        # 设置x轴和y轴刻度标签字体
        for label in ax.get_xticklabels():
            label.set_fontproperties(zh_font)
        
        for label in ax.get_yticklabels():
            label.set_fontproperties(zh_font)
        
        # 设置图例字体
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontproperties(zh_font)
        
        # 设置文本注释字体
        for artist in ax.get_children():
            if isinstance(artist, plt.Text):
                artist.set_fontproperties(zh_font)

def show_transformer_architecture():
    """
    展示Transformer架构的可视化页面
    """
    # 确保中文正确显示
    set_chinese_font()
    zh_font = force_set_chinese_font()
    
    st.header("Transformer架构可视化")
    
    st.markdown("""
    ### Transformer架构概述
    
    Transformer是一种基于自注意力机制的神经网络架构，由Google团队在2017年提出，已成为NLP领域的主流模型架构。
    不同于RNN或CNN，Transformer完全依赖注意力机制来捕捉输入和输出的全局依赖关系。
    
    Transformer主要由编码器(Encoder)和解码器(Decoder)两部分组成，下面我们将详细介绍其结构和工作原理。
    """)
    
    # 获取Transformer配置数据
    encoder_layers, decoder_layers, attention_heads, components = get_demo_data("transformer")
    
    # 显示整体架构图
    st.image("https://miro.medium.com/max/1400/1*BHzGVskWGS_3jEcYYi6miQ.png", 
             caption="Transformer架构图（来源：Attention is All You Need论文）")
    
    # 展示编码器和解码器结构
    st.subheader("编码器和解码器")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### 编码器 (Encoder)
        
        编码器由 **{encoder_layers}** 个相同的层堆叠而成，每一层包含：
        
        1. **多头自注意力机制** - 允许编码器关注输入序列的不同位置
        2. **前馈神经网络** - 包含两个线性变换和ReLU激活函数
        3. **残差连接和层归一化** - 帮助训练更深的网络
        
        编码器的输出是一个向量序列，包含输入序列中每个位置的表示信息。
        """)
        
        # 绘制简化的编码器结构
        fig1, ax1 = plt.subplots(figsize=(5, 8))
        
        # 背景色和文本颜色
        encoder_color = "#a8d5ba"
        text_color = "#000000"
        
        # 绘制编码器背景框
        encoder_rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, color=encoder_color, alpha=0.3)
        ax1.add_patch(encoder_rect)
        
        # 添加组件块
        components_heights = [0.15, 0.2, 0.25, 0.15]
        components_names = ["输入嵌入 + 位置编码", "多头自注意力", "前馈神经网络", "输出"]
        component_colors = ["#ffcccc", "#ccccff", "#ffffcc", "#ccffcc"]
        
        y_pos = 0.2
        for i, (height, name, color) in enumerate(zip(components_heights, components_names, component_colors)):
            rect = plt.Rectangle((0.2, y_pos), 0.6, height, fill=True, color=color, alpha=0.7)
            ax1.add_patch(rect)
            ax1.text(0.5, y_pos + height/2, name, ha='center', va='center', color=text_color, fontweight='bold', 
                    fontproperties=zh_font)
            
            # 如果不是最后一个组件，添加箭头
            if i < len(components_heights) - 1:
                ax1.arrow(0.5, y_pos + height, 0, 0.05, head_width=0.05, head_length=0.03, fc='black', ec='black')
            
            y_pos += height + 0.05
        
        # 标注编码器
        ax1.text(0.5, 0.95, f"Encoder ({encoder_layers} layers)", ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color)
        
        # 移除坐标轴
        ax1.axis('off')
        
        # 应用字体到所有文本元素
        apply_chinese_font_to_plot(fig1, zh_font)
        
        st.pyplot(fig1)
    
    with col2:
        st.markdown(f"""
        ### 解码器 (Decoder)
        
        解码器由 **{decoder_layers}** 个相同的层堆叠而成，每一层包含：
        
        1. **带掩码的多头自注意力机制** - 只能关注到当前及之前的位置
        2. **编码器-解码器多头注意力机制** - 关注编码器的输出
        3. **前馈神经网络** - 类似于编码器中的前馈网络
        4. **残差连接和层归一化** - 帮助梯度流动
        
        解码器的特点是采用自回归(auto-regressive)方式生成序列，每次生成一个词元。
        """)
        
        # 绘制简化的解码器结构
        fig2, ax2 = plt.subplots(figsize=(5, 8))
        
        # 背景色和文本颜色
        decoder_color = "#d5a8ba"
        text_color = "#000000"
        
        # 绘制解码器背景框
        decoder_rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=True, color=decoder_color, alpha=0.3)
        ax2.add_patch(decoder_rect)
        
        # 添加组件块
        components_heights = [0.1, 0.15, 0.15, 0.15, 0.1]
        components_names = ["输出嵌入 + 位置编码", "带掩码的多头自注意力", "编码器-解码器多头注意力", "前馈神经网络", "线性层 + Softmax"]
        component_colors = ["#ffcccc", "#ccccff", "#ddaaff", "#ffffcc", "#ccffcc"]
        
        y_pos = 0.2
        for i, (height, name, color) in enumerate(zip(components_heights, components_names, component_colors)):
            rect = plt.Rectangle((0.2, y_pos), 0.6, height, fill=True, color=color, alpha=0.7)
            ax2.add_patch(rect)
            ax2.text(0.5, y_pos + height/2, name, ha='center', va='center', color=text_color, fontweight='bold',
                    fontproperties=zh_font)
            
            # 如果不是最后一个组件，添加箭头
            if i < len(components_heights) - 1:
                ax2.arrow(0.5, y_pos + height, 0, 0.05, head_width=0.05, head_length=0.03, fc='black', ec='black')
            
            y_pos += height + 0.05
        
        # 标注解码器
        ax2.text(0.5, 0.95, f"Decoder ({decoder_layers} layers)", ha='center', va='center', 
                fontsize=12, fontweight='bold', color=text_color)
        
        # 移除坐标轴
        ax2.axis('off')
        
        # 应用字体到所有文本元素
        apply_chinese_font_to_plot(fig2, zh_font)
        
        st.pyplot(fig2)
    
    # 核心组件详细说明
    st.subheader("核心组件详解")
    
    component_to_show = st.selectbox("选择要详细了解的组件：", list(components.keys()))
    
    # 根据选择显示不同组件的详细信息
    st.markdown(f"### {component_to_show}")
    st.markdown(components[component_to_show])
    
    if component_to_show == "multi_head_attention":
        st.markdown("""
        多头注意力机制将查询(Q)、键(K)和值(V)线性投影到多个子空间，对每个子空间独立计算注意力，然后合并结果。
        
        数学表达式：
        
        $$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\text{head}_2, ..., \\text{head}_h)W^O$$
        
        其中：
        
        $$\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
        """)
        
        # 多头注意力的可视化图示
        st.image("https://miro.medium.com/max/1400/1*1K-VULoWLHWCrYPcbH_YYA.png",
                caption="多头注意力机制示意图")
    
    elif component_to_show == "positional_encoding":
        st.markdown("""
        位置编码用于为序列中的每个位置添加位置信息，使模型能够利用序列中词元的相对或绝对位置。
        
        Transformer使用正弦和余弦函数的组合来生成位置编码：
        
        $$PE_{(pos,2i)} = \\sin(pos/10000^{2i/d_{model}})$$
        $$PE_{(pos,2i+1)} = \\cos(pos/10000^{2i/d_{model}})$$
        
        这里pos是位置，i是维度索引。
        """)
        
        # 绘制位置编码热图
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # 生成位置编码矩阵
        d_model = 64
        max_len = 30
        
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # 绘制热图
        im = ax3.imshow(pe, cmap='viridis')
        fig3.colorbar(im, ax=ax3)
        
        # 手动设置标题和标签，确保使用正确的字体
        ax3.set_title("位置编码矩阵可视化", fontproperties=zh_font)
        ax3.set_xlabel("编码维度", fontproperties=zh_font)
        ax3.set_ylabel("序列位置", fontproperties=zh_font)
        
        # 应用字体到所有文本元素
        apply_chinese_font_to_plot(fig3, zh_font)
        
        st.pyplot(fig3)
    
    # 交互式Transformer模型演示
    st.subheader("Transformer参数交互式探索")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_encoder_layers = st.slider("编码器层数", 1, 12, encoder_layers)
    
    with col2:
        user_decoder_layers = st.slider("解码器层数", 1, 12, decoder_layers)
    
    with col3:
        user_attention_heads = st.slider("注意力头数", 1, 16, attention_heads)
    
    st.markdown(f"""
    ### 自定义Transformer配置
    
    您选择的配置：
    - 编码器层数: **{user_encoder_layers}**
    - 解码器层数: **{user_decoder_layers}**
    - 注意力头数: **{user_attention_heads}**
    
    #### 参数分析：
    
    1. **编码器层数**增加可以提高模型捕捉复杂模式的能力，但也会增加计算开销和过拟合风险。
    
    2. **解码器层数**增加可以提高生成质量，但同样会增加计算复杂度。
    
    3. **注意力头数**增加允许模型从不同的表示子空间学习信息，捕捉更丰富的特征关系。
    
    您的配置总参数量约为: **{calculate_param_count(user_encoder_layers, user_decoder_layers, user_attention_heads):,}**
    """)
    
    # 显示不同规模Transformer的应用
    st.subheader("不同规模Transformer的应用")
    
    st.markdown("""
    | 模型大小 | 参数数量 | 典型应用 |
    |---------|----------|---------|
    | 小型    | 10M-100M | 文本分类、情感分析、简单的生成任务 |
    | 中型    | 100M-1B  | 机器翻译、文本摘要、一般NLP任务 |
    | 大型    | 1B-10B   | 高质量文本生成、问答系统 |
    | 超大型  | >10B     | 跨领域理解、高质量创意内容生成、人类水平推理 |
    """)


def calculate_param_count(encoder_layers, decoder_layers, nhead, d_model=512, d_ff=2048, vocab_size=30000):
    """
    估算Transformer模型的参数数量
    
    Args:
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
        nhead: 注意力头数
        d_model: 模型维度
        d_ff: 前馈网络维度
        vocab_size: 词汇表大小
    
    Returns:
        param_count: 估计的参数数量
    """
    # 每个注意力头的参数量
    head_dim = d_model // nhead
    
    # 一个编码器层的参数量
    encoder_layer_params = (
        # 多头自注意力
        3 * d_model * d_model +  # Q, K, V投影
        d_model * d_model +      # 输出投影
        
        # 前馈网络
        d_model * d_ff +         # 第一个线性层
        d_ff * d_model +         # 第二个线性层
        
        # 层归一化参数
        4 * d_model              # 2层归一化，每层有缩放和偏置参数
    )
    
    # 一个解码器层的参数量
    decoder_layer_params = (
        # 第一个多头自注意力
        3 * d_model * d_model +  # Q, K, V投影
        d_model * d_model +      # 输出投影
        
        # 第二个多头注意力（编码器-解码器注意力）
        3 * d_model * d_model +  # Q, K, V投影
        d_model * d_model +      # 输出投影
        
        # 前馈网络
        d_model * d_ff +         # 第一个线性层
        d_ff * d_model +         # 第二个线性层
        
        # 层归一化参数
        6 * d_model              # 3层归一化，每层有缩放和偏置参数
    )
    
    # 嵌入层和输出层参数
    embedding_params = vocab_size * d_model  # 输入嵌入
    output_params = vocab_size * d_model     # 输出投影
    
    # 位置编码（如果使用学习的位置编码）
    positional_encoding_params = 0  # 使用固定的正弦位置编码，没有参数
    
    # 总参数量
    total_params = (
        encoder_layers * encoder_layer_params +
        decoder_layers * decoder_layer_params +
        embedding_params +
        output_params +
        positional_encoding_params
    )
    
    return int(total_params) 