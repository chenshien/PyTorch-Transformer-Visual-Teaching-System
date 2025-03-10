import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import pandas as pd
from utils.demo_data import get_demo_data
from utils.fix_chinese import set_chinese_font
import matplotlib.font_manager as fm
import os
import platform

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
            'C:/Windows/Fonts/simsun.ttc'   # 宋体
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                # 直接加载字体文件
                zh_font = fm.FontProperties(fname=font_path)
                plt.rcParams['font.sans-serif'] = [font_path] + plt.rcParams['font.sans-serif']
                return zh_font
    
    # 方法3: 尝试使用外部字体
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    emergency_font_path = os.path.join(script_dir, "utils", "emergency_font.ttf")
    if os.path.exists(emergency_font_path):
        zh_font = fm.FontProperties(fname=emergency_font_path)
        return zh_font
    
    # 如果所有方法都失败，返回默认字体
    return fm.FontProperties(family='sans-serif')

def show_training_visualization():
    """
    展示模型训练过程的可视化页面
    """
    # 确保中文正确显示
    set_chinese_font()
    zh_font = force_set_chinese_font()
    
    st.header("神经网络训练过程可视化")
    
    st.markdown("""
    ### 模型训练过程的可视化
    
    深度学习模型的训练是一个迭代优化的过程，通过可视化训练过程，我们可以更好地理解模型的学习情况和潜在问题。
    
    本页面将展示Transformer模型训练过程中的关键指标变化，帮助理解模型训练的动态过程。
    """)
    
    # 获取训练演示数据
    epochs, train_losses, val_losses, accuracies, weights_history = get_demo_data("training")
    
    # 选择可视化类型
    visualization_type = st.selectbox(
        "选择要查看的可视化类型:",
        ["损失曲线", "精度曲线", "权重分布变化", "交互式训练模拟", "学习率调整效果"]
    )
    
    if visualization_type == "损失曲线":
        show_loss_curves(epochs, train_losses, val_losses, zh_font)
    
    elif visualization_type == "精度曲线":
        show_accuracy_curve(epochs, accuracies, zh_font)
    
    elif visualization_type == "权重分布变化":
        show_weight_distribution(epochs, weights_history, zh_font)
    
    elif visualization_type == "交互式训练模拟":
        show_interactive_training(zh_font)
    
    elif visualization_type == "学习率调整效果":
        show_learning_rate_effects(zh_font)


def show_loss_curves(epochs, train_losses, val_losses, zh_font):
    """
    显示训练过程中的损失曲线
    """
    # 确保中文正确显示
    set_chinese_font()
    
    st.subheader("训练过程中的损失曲线")
    
    st.markdown("""
    ### 损失曲线分析
    
    损失曲线反映了模型在训练过程中的优化情况：
    
    - **训练损失**：模型在训练数据上的表现
    - **验证损失**：模型在验证数据上的表现
    
    理想情况下，两条曲线应该同时下降并最终收敛。如果验证损失不再下降甚至上升，而训练损失继续下降，则可能发生了过拟合。
    """)
    
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, 'b-', label='训练损失')
    ax.plot(epochs, val_losses, 'r-', label='验证损失')
    ax.set_title('训练和验证损失', fontproperties=zh_font)
    ax.set_xlabel('训练轮次', fontproperties=zh_font)
    ax.set_ylabel('损失值', fontproperties=zh_font)
    ax.legend(prop=zh_font)
    ax.grid(True)
    
    # 在关键点添加标注
    min_val_loss_idx = np.argmin(val_losses)
    ax.annotate(f'最低验证损失: {val_losses[min_val_loss_idx]:.3f}', 
                xy=(epochs[min_val_loss_idx], val_losses[min_val_loss_idx]),
                xytext=(epochs[min_val_loss_idx]+1, val_losses[min_val_loss_idx]+0.2),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontproperties=zh_font)
    
    st.pyplot(fig)
    
    # 添加一些随训练进度变化的观察结果
    progress_df = pd.DataFrame({
        '轮次': epochs,
        '训练损失': [f"{loss:.4f}" for loss in train_losses],
        '验证损失': [f"{loss:.4f}" for loss in val_losses],
        '训练-验证差异': [f"{train-val:.4f}" for train, val in zip(train_losses, val_losses)]
    })
    
    st.dataframe(progress_df)
    
    # 添加过拟合检测
    overfit_point = None
    for i in range(3, len(epochs)):
        if all(val_losses[i-j] > val_losses[i-j-1] for j in range(3)) and all(train_losses[i-j] < train_losses[i-j-1] for j in range(3)):
            overfit_point = i-3
            break
    
    if overfit_point:
        st.warning(f"在第 {epochs[overfit_point]} 轮后可能出现过拟合。此时验证损失开始上升，而训练损失继续下降。")
        st.markdown("""
        ### 过拟合解决方案：
        
        1. **提前停止训练** - 在验证损失开始上升前停止训练
        2. **增加正则化** - 使用L1/L2正则化或Dropout
        3. **增加训练数据** - 收集更多训练样本或使用数据增强
        4. **简化模型** - 减少模型参数或层数
        """)
    else:
        st.success("模型训练良好，未检测到明显的过拟合现象。")


def show_accuracy_curve(epochs, accuracies, zh_font):
    """
    显示训练过程中的准确率曲线
    """
    # 确保中文正确显示
    set_chinese_font()
    
    st.subheader("训练过程中的准确率曲线")
    
    st.markdown("""
    ### 准确率变化分析
    
    准确率曲线反映了模型在分类任务上的表现，它通常随着训练的进行而提高，最终趋于稳定。
    """)
    
    # 绘制准确率曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, accuracies, 'g-', marker='o')
    ax.set_title('验证准确率', fontproperties=zh_font)
    ax.set_xlabel('训练轮次', fontproperties=zh_font)
    ax.set_ylabel('准确率', fontproperties=zh_font)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # 标记最高准确率
    max_acc_idx = np.argmax(accuracies)
    ax.annotate(f'最高准确率: {accuracies[max_acc_idx]:.3f}', 
                xy=(epochs[max_acc_idx], accuracies[max_acc_idx]),
                xytext=(epochs[max_acc_idx]-2, accuracies[max_acc_idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontproperties=zh_font)
    
    st.pyplot(fig)
    
    # 添加准确率提升的阶段分析
    if len(epochs) > 5:
        early_stage = np.mean(np.diff(accuracies[:5]))
        mid_stage = np.mean(np.diff(accuracies[5:15])) if len(epochs) > 15 else np.mean(np.diff(accuracies[5:]))
        late_stage = np.mean(np.diff(accuracies[15:])) if len(epochs) > 15 else 0
        
        st.markdown("### 准确率提升阶段分析")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("早期阶段提升率", f"{early_stage:.3f}")
        with col2:
            st.metric("中期阶段提升率", f"{mid_stage:.3f}")
        with col3:
            st.metric("后期阶段提升率", f"{late_stage:.3f}")
        
        if early_stage > mid_stage > late_stage:
            st.info("模型学习呈现典型的先快后慢模式，早期学习速度最快，后期趋于饱和。")
        elif mid_stage > early_stage:
            st.info("模型在中期学习速度最快，可能是学习率调整或更复杂特征的学习所致。")


def show_weight_distribution(epochs, weights_history, zh_font):
    """
    显示训练过程中的权重分布变化
    """
    # 确保中文正确显示
    set_chinese_font()
    
    st.subheader("模型权重分布的变化")
    
    st.markdown("""
    ### 权重分布变化分析
    
    权重分布的变化反映了模型参数在训练过程中的调整情况，通过观察权重的变化，可以了解模型的学习过程。
    """)
    
    # 创建权重变化图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for weight_name, weight_values in weights_history.items():
        ax.plot(epochs, weight_values, label=weight_name)
    
    ax.set_title('权重变化趋势', fontproperties=zh_font)
    ax.set_xlabel('训练轮次', fontproperties=zh_font)
    ax.set_ylabel('权重值', fontproperties=zh_font)
    ax.legend(prop=zh_font)
    ax.grid(True)
    
    st.pyplot(fig)
    
    # 添加权重分布直方图
    epoch_to_show = st.slider('选择要查看的训练轮次:', min_value=1, max_value=len(epochs), value=len(epochs)//2)
    epoch_idx = epoch_to_show - 1
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 为每个权重生成随机分布数据（模拟真实权重分布）
    np.random.seed(42 + epoch_idx)
    for weight_name, weight_values in weights_history.items():
        # 生成围绕当前权重值的正态分布
        weight_distribution = np.random.normal(weight_values[epoch_idx], 0.1, 1000)
        ax2.hist(weight_distribution, bins=30, alpha=0.5, label=weight_name)
    
    ax2.set_title(f'第 {epoch_to_show} 轮训练后的权重分布', fontproperties=zh_font)
    ax2.set_xlabel('权重值', fontproperties=zh_font)
    ax2.set_ylabel('频率', fontproperties=zh_font)
    ax2.legend(prop=zh_font)
    
    st.pyplot(fig2)
    
    st.markdown("""
    ### 权重分布的意义
    
    - **集中在0附近的钟形分布** - 表示模型参数经过良好的正则化
    - **分散的多峰分布** - 可能表示不同特征学习到不同程度的特征
    - **极端值过多** - 可能表示梯度爆炸问题
    
    通常情况下，一个训练良好的模型权重应该呈现均值接近0、方差适中的分布。
    """)


def show_interactive_training(zh_font):
    """
    提供交互式的模型训练模拟
    """
    # 确保中文正确显示
    set_chinese_font()
    
    st.subheader("交互式训练模拟")
    
    st.markdown("""
    ### 模拟神经网络训练过程
    
    这个交互式演示模拟了一个简单神经网络的训练过程，您可以调整以下参数来观察它们对训练过程的影响。
    """)
    
    # 模型参数控制
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider("学习率", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        batch_size = st.slider("批次大小", 8, 128, 32, 8)
    
    with col2:
        hidden_size = st.slider("隐藏层大小", 10, 100, 50, 10)
        dropout_rate = st.slider("Dropout比率", 0.0, 0.5, 0.2, 0.1, format="%.1f")
    
    # 训练设置
    max_epochs = st.slider("训练轮次", 10, 100, 30, 5)
    
    # 模拟训练按钮
    if st.button("开始模拟训练"):
        with st.spinner("模拟训练中..."):
            # 模拟训练过程
            train_losses, val_losses, train_accs, val_accs = simulate_training(
                learning_rate, batch_size, hidden_size, dropout_rate, max_epochs
            )
            
            # 显示训练进度
            progress_bar = st.progress(0)
            epochs_range = list(range(1, max_epochs + 1))
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 初始化线条
            train_loss_line, = ax1.plot([], [], 'b-', label='训练损失')
            val_loss_line, = ax1.plot([], [], 'r-', label='验证损失')
            ax1.set_xlim(0, max_epochs)
            ax1.set_ylim(0, max(np.max(train_losses), np.max(val_losses)) * 1.1)
            ax1.set_title('损失曲线', fontproperties=zh_font)
            ax1.set_xlabel('轮次', fontproperties=zh_font)
            ax1.set_ylabel('损失', fontproperties=zh_font)
            ax1.legend(prop=zh_font)
            ax1.grid(True)
            
            train_acc_line, = ax2.plot([], [], 'b-', label='训练准确率')
            val_acc_line, = ax2.plot([], [], 'r-', label='验证准确率')
            ax2.set_xlim(0, max_epochs)
            ax2.set_ylim(0, 1.0)
            ax2.set_title('准确率曲线', fontproperties=zh_font)
            ax2.set_xlabel('轮次', fontproperties=zh_font)
            ax2.set_ylabel('准确率', fontproperties=zh_font)
            ax2.legend(prop=zh_font)
            ax2.grid(True)
            
            chart = st.pyplot(fig)
            
            # 动态更新图表
            for i in range(max_epochs):
                # 更新数据
                x_data = epochs_range[:i+1]
                train_loss_line.set_data(x_data, train_losses[:i+1])
                val_loss_line.set_data(x_data, val_losses[:i+1])
                train_acc_line.set_data(x_data, train_accs[:i+1])
                val_acc_line.set_data(x_data, val_accs[:i+1])
                
                # 重绘图表
                chart.pyplot(fig)
                
                # 更新进度条
                progress_bar.progress((i + 1) / max_epochs)
                
                # 短暂暂停
                time.sleep(0.1)
            
            # 分析训练结果
            min_val_loss = min(val_losses)
            min_val_loss_epoch = np.argmin(val_losses) + 1
            max_val_acc = max(val_accs)
            max_val_acc_epoch = np.argmax(val_accs) + 1
            
            # 展示训练结果
            st.success("模拟训练完成！")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最终训练损失", f"{train_losses[-1]:.4f}")
            col2.metric("最低验证损失", f"{min_val_loss:.4f} (轮次 {min_val_loss_epoch})")
            col3.metric("最终训练准确率", f"{train_accs[-1]:.2%}")
            col4.metric("最高验证准确率", f"{max_val_acc:.2%} (轮次 {max_val_acc_epoch})")
            
            # 训练分析
            st.subheader("训练结果分析")
            
            # 过拟合分析
            overfit_metric = (train_accs[-1] - val_accs[-1]) / train_accs[-1]
            
            if overfit_metric > 0.2:
                st.warning(f"检测到明显的过拟合现象。训练准确率与验证准确率差距为 {overfit_metric:.2%}。")
                st.markdown("""
                ### 建议：
                - 增加Dropout率
                - 添加L2正则化
                - 减小网络规模或提前停止
                """)
            elif overfit_metric > 0.05:
                st.warning(f"检测到轻微的过拟合现象。训练准确率与验证准确率差距为 {overfit_metric:.2%}。")
            else:
                st.success("模型拟合良好，没有明显的过拟合。")
            
            # 学习率分析
            if min_val_loss_epoch > max_epochs * 0.8:
                st.info("验证损失在训练后期仍在下降，可能需要更多的训练轮次或更大的学习率。")
            elif min_val_loss_epoch < max_epochs * 0.2:
                st.info("验证损失在训练早期就达到最低，可能学习率过大或模型容量不足。")


def simulate_training(learning_rate, batch_size, hidden_size, dropout_rate, max_epochs):
    """
    模拟简单神经网络的训练过程
    
    Args:
        learning_rate: 学习率
        batch_size: 批次大小
        hidden_size: 隐藏层大小
        dropout_rate: Dropout比率
        max_epochs: 最大训练轮次
    
    Returns:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率
        val_accs: 验证准确率
    """
    # 使用确定性随机种子以保证结果的一致性
    np.random.seed(42)
    
    # 模拟一个具有噪声的训练曲线
    # 基础曲线是一个指数衰减函数
    base_train_loss = 2.0 * np.exp(-0.15 * np.array(range(1, max_epochs + 1)))
    base_val_loss = 2.0 * np.exp(-0.12 * np.array(range(1, max_epochs + 1)))
    
    # 添加噪声
    train_losses = base_train_loss + 0.1 * np.random.randn(max_epochs)
    val_losses = base_val_loss + 0.15 * np.random.randn(max_epochs)
    
    # 应用学习率影响
    lr_factor = 1.0 - 0.5 * (learning_rate - 0.01) / 0.09  # 标准化到[0.5, 1.5]范围
    train_losses *= lr_factor
    val_losses *= lr_factor
    
    # 应用batch_size影响
    batch_factor = 0.8 + 0.4 * (batch_size - 8) / 120  # 标准化到[0.8, 1.2]范围
    train_losses *= batch_factor
    val_losses *= batch_factor ** 0.8  # 对验证损失影响较小
    
    # 应用dropout影响
    dropout_factor = 1.0 + dropout_rate  # dropout越大，初始损失越大
    train_losses *= dropout_factor
    
    # 模拟过拟合影响
    overfit_start = int(max_epochs * (0.5 - dropout_rate))  # dropout越大，过拟合开始越晚
    
    for i in range(overfit_start, max_epochs):
        decay_factor = (i - overfit_start) / (max_epochs - overfit_start)
        train_losses[i] *= (0.95 - 0.3 * decay_factor)  # 训练损失继续下降
        val_losses[i] *= (1.0 + 0.5 * decay_factor * (1 - dropout_rate * 2))  # 验证损失上升，dropout可以缓解
    
    # 确保损失值合理
    train_losses = np.clip(train_losses, 0.1, 2.0)
    val_losses = np.clip(val_losses, 0.1, 2.0)
    
    # 根据损失计算准确率
    train_accs = 1.0 - train_losses / 2.5
    val_accs = 1.0 - val_losses / 2.5
    
    # 确保准确率值合理
    train_accs = np.clip(train_accs, 0.5, 0.99)
    val_accs = np.clip(val_accs, 0.5, 0.99)
    
    return train_losses, val_losses, train_accs, val_accs


def show_learning_rate_effects(zh_font):
    """
    显示不同学习率的效果
    """
    # 确保中文正确显示
    set_chinese_font()
    
    st.subheader("学习率对训练过程的影响")
    
    st.markdown("""
    ### 学习率的重要性
    
    学习率是深度学习中最重要的超参数之一，它决定了模型参数在每次更新时的步长大小：
    
    - **过高的学习率**：可能导致训练不稳定或发散
    - **过低的学习率**：收敛速度慢，可能陷入局部最小值
    - **合适的学习率**：能够在合理时间内达到较好的性能
    
    下面的可视化展示了不同学习率对训练过程的影响。
    """)
    
    # 模拟不同学习率的训练过程
    epochs = np.arange(1, 31)
    
    # 不同学习率下的损失曲线
    lr_too_small = 2.0 * np.exp(-0.05 * epochs) + 0.1 * np.random.randn(30)
    lr_good = 2.0 * np.exp(-0.15 * epochs) + 0.1 * np.random.randn(30)
    lr_too_large = np.ones(30) * 0.5 + 0.4 * np.sin(epochs) + 0.2 * np.random.randn(30)
    
    # 确保损失值合理
    lr_too_small = np.clip(lr_too_small, 0.1, 2.5)
    lr_good = np.clip(lr_good, 0.1, 2.5)
    lr_too_large = np.clip(lr_too_large, 0.1, 2.5)
    
    # 绘制损失曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lr_too_small, 'b-', label='学习率过小 (0.001)')
    ax.plot(epochs, lr_good, 'g-', label='合适学习率 (0.01)')
    ax.plot(epochs, lr_too_large, 'r-', label='学习率过大 (0.1)')
    ax.set_title('不同学习率下的训练损失', fontproperties=zh_font)
    ax.set_xlabel('训练轮次', fontproperties=zh_font)
    ax.set_ylabel('损失', fontproperties=zh_font)
    ax.legend(prop=zh_font)
    ax.grid(True)
    
    st.pyplot(fig)
    
    # 学习率策略
    st.markdown("""
    ### 学习率调整策略
    
    为了获得更好的训练效果，通常会采用以下学习率调整策略：
    
    1. **学习率衰减**：随着训练进行，逐渐减小学习率
    2. **学习率预热**：从小学习率开始，逐渐增加到目标值，然后再衰减
    3. **周期性学习率**：学习率在一定范围内周期性变化
    4. **自适应学习率**：根据梯度信息自动调整学习率(Adam, RMSprop等优化器)
    """)
    
    # 学习率调整策略可视化
    lr_strategy = st.selectbox(
        "选择学习率调整策略:",
        ["固定学习率", "阶梯衰减", "指数衰减", "余弦衰减", "周期性变化"]
    )
    
    x = np.arange(0, 100)
    
    if lr_strategy == "固定学习率":
        y = np.ones_like(x) * 0.01
        title = "固定学习率"
    elif lr_strategy == "阶梯衰减":
        y = np.ones_like(x) * 0.01
        y[30:60] = 0.005
        y[60:] = 0.001
        title = "阶梯衰减学习率"
    elif lr_strategy == "指数衰减":
        y = 0.01 * np.exp(-0.02 * x)
        title = "指数衰减学习率"
    elif lr_strategy == "余弦衰减":
        y = 0.001 + 0.009 * (1 + np.cos(np.pi * x / 100)) / 2
        title = "余弦衰减学习率"
    elif lr_strategy == "周期性变化":
        y = 0.001 + 0.009 * 0.5 * (1 + np.cos(np.pi * np.mod(x, 20) / 10))
        title = "周期性变化学习率"
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x, y, 'b-')
    ax2.set_title(title, fontproperties=zh_font)
    ax2.set_xlabel('训练步数', fontproperties=zh_font)
    ax2.set_ylabel('学习率', fontproperties=zh_font)
    ax2.grid(True)
    
    st.pyplot(fig2)
    
    # 使用建议
    st.markdown("""
    ### 学习率选择建议
    
    1. **模型初始化后**：先尝试较小的学习率(如0.001)，观察是否收敛
    2. **如果收敛太慢**：尝试增加学习率
    3. **如果训练不稳定**：降低学习率或使用Adam等自适应优化器
    4. **对于大模型**：通常采用学习率预热加余弦衰减的策略
    5. **参考经验值**：
       - 一般CNN: 0.01-0.001
       - Transformer: 0.0001-0.00001
       - 微调预训练模型: 0.00005-0.000001
    """)
    
    # 学习率查找器介绍
    st.info("""
    **学习率查找器(Learning Rate Finder)**是一种自动寻找最佳初始学习率的技术。它从极小的学习率开始，逐渐增加，直到损失开始急剧上升。理想的学习率通常是损失开始下降最快的点。
    
    在PyTorch中，可以使用[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)或[fastai](https://docs.fast.ai/)库中的学习率查找器功能。
    """) 