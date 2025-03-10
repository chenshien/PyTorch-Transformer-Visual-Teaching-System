import torch
import numpy as np
import pandas as pd
import os

def get_demo_data(demo_type="attention"):
    """
    获取演示用的数据
    
    Args:
        demo_type: 数据类型，可选值为 "attention", "training", "transformer"
    
    Returns:
        根据demo_type返回不同类型的数据
    """
    if demo_type == "attention":
        return get_attention_demo_data()
    elif demo_type == "training":
        return get_training_demo_data()
    elif demo_type == "transformer":
        return get_transformer_demo_data()
    else:
        raise ValueError(f"不支持的演示类型: {demo_type}")


def get_attention_demo_data():
    """
    获取自注意力机制演示用数据
    
    Returns:
        tokens: 示例句子中的词元列表
        query: 示例查询向量
        key: 示例键向量  
        value: 示例值向量
    """
    # 示例句子
    tokens = ["我", "喜欢", "深度", "学习", "和", "自然", "语言", "处理"]
    
    # 模拟词元嵌入向量（实际应用中应该是从预训练模型中获取）
    # 为简化演示，我们使用4维向量
    d_model = 4
    
    # 生成随机嵌入向量
    np.random.seed(42)  # 设置随机种子以确保结果可重现
    embeddings = np.random.randn(len(tokens), d_model)
    
    # 转换为PyTorch张量
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # 对于演示，我们使用相同的嵌入作为q、k、v
    query = embeddings
    key = embeddings
    value = embeddings
    
    return tokens, query, key, value


def get_training_demo_data():
    """
    获取模型训练可视化演示用数据
    
    Returns:
        epochs: 训练轮次
        train_losses: 训练损失
        val_losses: 验证损失
        accuracies: 准确率
        weights_history: 权重变化历史
    """
    # 模拟训练过程数据
    epochs = list(range(1, 21))
    
    # 模拟损失下降曲线
    np.random.seed(42)
    train_losses = 1.5 * np.exp(-0.1 * np.array(epochs)) + 0.1 * np.random.rand(len(epochs))
    val_losses = 1.8 * np.exp(-0.08 * np.array(epochs)) + 0.15 * np.random.rand(len(epochs))
    
    # 模拟准确率上升曲线
    base_acc = 0.5 + 0.4 * (1 - np.exp(-0.15 * np.array(epochs)))
    accuracies = base_acc + 0.05 * np.random.rand(len(epochs))
    accuracies = np.clip(accuracies, 0, 1)  # 确保准确率在[0,1]范围内
    
    # 模拟权重变化
    # 假设我们跟踪3个主要权重
    w1_history = 0.2 * np.sin(np.array(epochs) * 0.5) + 0.5
    w2_history = -0.3 * np.cos(np.array(epochs) * 0.3) + 0.1
    w3_history = 0.5 * np.exp(-0.1 * np.array(epochs)) - 0.2
    
    weights_history = {
        'weight1': w1_history,
        'weight2': w2_history,
        'weight3': w3_history
    }
    
    return epochs, train_losses, val_losses, accuracies, weights_history


def get_transformer_demo_data():
    """
    获取Transformer架构演示用数据
    
    Returns:
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
        attention_heads: 注意力头数量
        components: Transformer组件列表
    """
    # Transformer基本配置
    encoder_layers = 6
    decoder_layers = 6
    attention_heads = 8
    
    # Transformer主要组件
    components = {
        "input_embedding": "将输入词元转换为固定维度的向量表示",
        "positional_encoding": "为序列中的每个位置添加位置信息",
        "multi_head_attention": "允许模型关注序列中不同位置的信息",
        "feed_forward": "对每个位置应用相同的前馈神经网络",
        "layer_norm": "对每层输出进行归一化处理",
        "residual_connection": "通过残差连接帮助梯度流动和信息传递",
        "softmax": "将输出转换为概率分布"
    }
    
    return encoder_layers, decoder_layers, attention_heads, components


def save_example_data(save_dir="examples"):
    """生成并保存示例数据文件"""
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建示例句子数据
    sentences = [
        "深度学习已经成为人工智能领域的核心技术。",
        "Transformer模型在自然语言处理任务中表现出色。",
        "注意力机制使模型能够关注输入序列中的重要部分。",
        "PyTorch是一个灵活的深度学习框架。",
        "自监督学习减少了对标注数据的依赖。"
    ]
    
    # 创建示例数据集
    df = pd.DataFrame({
        'id': range(1, len(sentences) + 1),
        'sentence': sentences,
        'length': [len(s) for s in sentences]
    })
    
    # 保存为CSV文件
    df.to_csv(os.path.join(save_dir, 'sample_sentences.csv'), index=False, encoding='utf-8')
    
    # 创建示例注意力权重
    np.random.seed(42)
    attention_weights = np.random.rand(8, 10, 10)  # 8个头，10x10的注意力矩阵
    attention_weights = attention_weights / attention_weights.sum(axis=2, keepdims=True)  # 归一化
    
    # 保存为numpy文件
    np.save(os.path.join(save_dir, 'sample_attention.npy'), attention_weights)
    
    print(f"示例数据已保存到 {save_dir} 目录")


if __name__ == "__main__":
    # 如果直接运行此脚本，生成并保存示例数据
    save_example_data() 