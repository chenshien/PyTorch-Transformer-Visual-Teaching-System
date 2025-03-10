import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.simple_transformer import SimpleMultiHeadAttention
from utils.demo_data import get_demo_data
from utils.fix_chinese import set_chinese_font
import io
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

def visualize_attention():
    """
    自注意力机制可视化页面
    """
    # 确保中文正确显示
    set_chinese_font()
    zh_font = force_set_chinese_font()
    
    st.header("自注意力机制(Self-Attention)可视化")
    
    st.markdown("""
    ### 什么是自注意力机制？
    
    自注意力机制是Transformer架构的核心组件之一，它允许模型在处理序列时关注序列中的不同部分，并捕捉元素之间的依赖关系。
    
    ### 计算过程：
    
    1. 将输入向量转换为查询(Query)、键(Key)和值(Value)向量
    2. 计算Query与每个Key的点积，获得注意力分数
    3. 对注意力分数进行缩放并应用Softmax得到注意力权重
    4. 将注意力权重与Value向量加权求和得到输出
    
    $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
    """)
    
    # 获取示例数据
    tokens, query, key, value = get_demo_data("attention")
    
    # 交互式自注意力计算
    st.subheader("交互式自注意力演示")
    
    # 添加一个侧边栏参数控制
    st.sidebar.markdown("### 自注意力参数")
    scale_factor = st.sidebar.slider("缩放因子", 0.1, 2.0, 1.0, 0.1)
    temperature = st.sidebar.slider("温度系数 (softmax温度)", 0.1, 5.0, 1.0, 0.1)
    
    # 计算注意力分数和权重
    with torch.no_grad():
        # 计算注意力分数 (Q * K^T)
        attention_scores = torch.matmul(query, key.transpose(0, 1))
        
        # 缩放
        attention_scores = attention_scores / (query.size(-1) ** 0.5 * scale_factor)
        
        # 应用温度
        attention_scores = attention_scores / temperature
        
        # Softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        context_vectors = torch.matmul(attention_weights, value)
    
    # 展示中间计算结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 注意力分数 (Attention Scores)")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        
        # 使用seaborn的heatmap，确保字体正确设置
        sns.heatmap(attention_scores.numpy(), annot=True, fmt=".2f", cmap="YlGnBu", 
                   xticklabels=tokens, yticklabels=tokens, ax=ax1)
        
        # 手动设置标题和标签，确保使用正确的字体
        ax1.set_title("注意力分数", fontproperties=zh_font)
        ax1.set_xlabel("词元", fontproperties=zh_font)
        ax1.set_ylabel("词元", fontproperties=zh_font)
        
        # 确保所有标签使用中文字体
        for label in ax1.get_xticklabels():
            label.set_fontproperties(zh_font)
        for label in ax1.get_yticklabels():
            label.set_fontproperties(zh_font)
            
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.markdown("#### 注意力权重 (Attention Weights)")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        
        # 使用seaborn的heatmap，确保字体正确设置
        sns.heatmap(attention_weights.numpy(), annot=True, fmt=".2f", cmap="YlOrRd", 
                   xticklabels=tokens, yticklabels=tokens, ax=ax2)
        
        # 手动设置标题和标签，确保使用正确的字体
        ax2.set_title("注意力权重", fontproperties=zh_font)
        ax2.set_xlabel("词元", fontproperties=zh_font)
        ax2.set_ylabel("词元", fontproperties=zh_font)
        
        # 确保所有标签使用中文字体
        for label in ax2.get_xticklabels():
            label.set_fontproperties(zh_font)
        for label in ax2.get_yticklabels():
            label.set_fontproperties(zh_font)
            
        plt.tight_layout()
        st.pyplot(fig2)
    
    # 展示词元关系图
    st.subheader("词元间关系可视化")
    
    # 选择要可视化的词元
    selected_token = st.selectbox("选择一个词元查看它与其他词元的关注度:", tokens)
    selected_idx = tokens.index(selected_token)
    
    # 为选定词元创建条形图
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    token_weights = attention_weights[selected_idx].numpy()
    bars = ax3.barh(tokens, token_weights, color='skyblue')
    
    # 为最高权重添加不同颜色
    max_idx = np.argmax(token_weights)
    bars[max_idx].set_color('coral')
    
    # 设置中文标题和标签
    ax3.set_title(f"'{selected_token}' 对其他词元的注意力分配", fontproperties=zh_font)
    ax3.set_xlabel("注意力权重", fontproperties=zh_font)
    
    # 确保所有y轴标签使用中文字体
    ax3.tick_params(axis='y', labelsize=10)
    for label in ax3.get_yticklabels():
        label.set_fontproperties(zh_font)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # 多头注意力机制
    st.subheader("多头注意力机制 (Multi-Head Attention)")
    
    st.markdown("""
    多头注意力机制通过多个"注意力头"并行处理信息，每个头关注输入的不同部分，然后将结果合并。
    
    这使模型能够同时捕捉不同类型的依赖关系，如语法结构、语义关联等。
    """)
    
    # 简单演示多头注意力
    num_heads = st.slider("注意力头数量", 1, 8, 4)
    
    # 创建一个多头注意力图表
    fig4, axes = plt.subplots(2, int(np.ceil(num_heads/2)), figsize=(12, 8))
    axes = axes.flatten()
    
    # 生成每个注意力头的随机权重（实际情况下这些应该是模型学习的）
    np.random.seed(42)
    for i in range(num_heads):
        head_weights = np.random.rand(len(tokens), len(tokens))
        head_weights = head_weights / head_weights.sum(axis=1, keepdims=True)  # 归一化
        
        # 使用seaborn绘制热图
        sns.heatmap(head_weights, annot=True, fmt=".2f", cmap=f"Blues", 
                   xticklabels=tokens, yticklabels=tokens, ax=axes[i])
        
        # 设置标题和确保标签使用中文字体
        axes[i].set_title(f"注意力头 {i+1}", fontproperties=zh_font)
        
        for label in axes[i].get_xticklabels():
            label.set_fontproperties(zh_font)
        for label in axes[i].get_yticklabels():
            label.set_fontproperties(zh_font)
    
    # 隐藏未使用的子图
    for i in range(num_heads, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    st.pyplot(fig4)
    
    # 模型演示选项
    st.subheader("自定义示例")
    
    # 让用户输入自己的句子
    user_input = st.text_area("输入一个短句子进行注意力可视化:", "自然语言处理是人工智能的重要分支")
    
    if st.button("计算注意力分布"):
        # 分词（这里简化为字符级别）
        input_tokens = list(user_input)
        
        if len(input_tokens) > 20:
            st.warning("句子较长，仅显示前20个词元")
            input_tokens = input_tokens[:20]
        
        # 随机生成一个注意力矩阵（实际应用中应使用真实模型）
        n = len(input_tokens)
        random_attention = np.random.rand(n, n)
        # 归一化
        random_attention = random_attention / random_attention.sum(axis=1, keepdims=True)
        
        # 绘制注意力热图
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        
        # 确保使用中文字体
        sns.heatmap(random_attention, annot=True, fmt=".2f", cmap="YlOrRd", 
                   xticklabels=input_tokens, yticklabels=input_tokens, ax=ax5)
        
        # 手动设置标题和标签，确保使用正确的字体
        ax5.set_title("生成的注意力分布", fontproperties=zh_font)
        
        # 确保所有标签使用中文字体
        for label in ax5.get_xticklabels():
            label.set_fontproperties(zh_font)
        for label in ax5.get_yticklabels():
            label.set_fontproperties(zh_font)
        
        # 应用中文字体到所有文本元素
        apply_chinese_font_to_plot(fig5, zh_font)
            
        plt.tight_layout()
        st.pyplot(fig5)
        
        st.info("注：这是一个随机生成的注意力分布用于演示。在实际的Transformer模型中，注意力分布是通过学习得到的，能够捕捉词元之间的语义关系。")


def plot_attention_on_text(tokens, attention_weights):
    """
    绘制文本上的注意力权重可视化
    
    Args:
        tokens: 词元列表
        attention_weights: 注意力权重矩阵
    
    Returns:
        fig: Matplotlib图像对象
    """
    # 确保中文正确显示
    set_chinese_font()
    zh_font = force_set_chinese_font()
    
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(n*0.8, n*0.8))
    
    im = ax.imshow(attention_weights, cmap="YlOrRd")
    
    # 添加坐标轴标签
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)
    
    # 将x轴标签旋转90度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 设置中文字体
    for label in ax.get_xticklabels():
        label.set_fontproperties(zh_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(zh_font)
    
    # 在每个单元格上添加文本
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                          ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white")
    
    ax.set_title("注意力权重矩阵", fontproperties=zh_font)
    fig.tight_layout()
    
    # 应用中文字体到所有文本元素
    apply_chinese_font_to_plot(fig, zh_font)
    
    return fig 