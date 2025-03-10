import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import matplotlib.font_manager as fm
import platform

# 导入中文支持
from utils.fix_chinese import init_chinese_support, configure_streamlit_for_chinese, plot_with_bitmap_text

# 导入项目模块
from visualizations.attention_visualization import visualize_attention, force_set_chinese_font
from visualizations.transformer_architecture import show_transformer_architecture
from visualizations.training_visualization import show_training_visualization
from models.simple_transformer import SimpleTransformer
from utils.demo_data import get_demo_data

# 设置中文支持
is_font_set = init_chinese_support()

# 注册所有可用的中文字体
def register_all_fonts():
    if platform.system() == 'Windows':
        font_dirs = [
            "C:/Windows/Fonts",  # 标准Windows字体目录
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts")  # 用户字体目录
        ]
        chinese_fonts = [
            'msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc', 
            'simkai.ttf', 'simfang.ttf'
        ]
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file in chinese_fonts:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        try:
                            fm.fontManager.addfont(font_path)
                            print(f"全局注册字体: {font_path}")
                        except:
                            pass

# 注册所有字体以确保中文显示正常
register_all_fonts()

# 强制预加载并设置中文字体
zh_font = force_set_chinese_font()

# 设置页面配置（使用中文优化的配置）
configure_streamlit_for_chinese()

# 如果没有找到合适的中文字体，显示警告
if not is_font_set:
    st.warning("""
    ⚠️ 警告：未能找到合适的中文字体，图表中的中文可能显示为方块。
    
    解决方案：
    1. 安装中文字体，如"微软雅黑"、"宋体"或"文泉驿微米黑"
    2. 重新启动应用程序
    
    更多详情请查看README.md文件中的"中文显示问题解决方案"部分。
    """)

# 添加页面顶部的字体控制面板
with st.sidebar.expander("高级中文字体设置"):
    st.markdown("如果中文仍然显示不正常，请点击以下按钮尝试从网络下载紧急字体：")
    if st.button("下载紧急中文字体"):
        with st.spinner("正在下载紧急中文字体..."):
            try:
                import urllib.request
                noto_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
                utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils")
                noto_path = os.path.join(utils_dir, "emergency_font.ttf")
                
                # 下载字体
                urllib.request.urlretrieve(noto_url, noto_path)
                
                if os.path.exists(noto_path):
                    st.success("已成功下载紧急中文字体，请刷新页面以应用更改")
                    # 注册新字体
                    try:
                        fm.fontManager.addfont(noto_path)
                    except:
                        pass
                else:
                    st.error("下载字体失败")
            except Exception as e:
                st.error(f"下载字体时出错: {e}")
                st.info("请尝试手动安装中文字体，参考README.md中的说明")

# 添加自动滚动到顶部的功能
def auto_scroll_to_top():
    # 使用HTML和JavaScript实现自动滚动到顶部
    js_code = """
    <script>
        // 自动滚动到顶部
        window.scrollTo(0, 0);
        
        // 监听会话状态变化，当页面切换时滚动到顶部
        window.addEventListener('load', function() {
            // 获取当前会话状态
            const observer = new MutationObserver(function(mutations) {
                window.scrollTo(0, 0);
            });
            
            // 监视文档变化
            observer.observe(document.body, {childList: true, subtree: true});
        });
    </script>
    """
    st.components.v1.html(js_code, height=0)

# 调用自动滚动功能
auto_scroll_to_top()

# 侧边栏导航
st.sidebar.title("PyTorch与Transformer可视化教学系统")
st.sidebar.markdown("---")

# 使用session_state跟踪页面变化
if 'previous_page' not in st.session_state:
    st.session_state.previous_page = None

# 定义页面选择函数
def on_page_change():
    # 如果页面发生变化，记录下来
    if st.session_state.page != st.session_state.previous_page:
        st.session_state.previous_page = st.session_state.page
        # 页面变化时触发的操作可以放在这里

# 使用session_state跟踪当前页面
page = st.sidebar.radio(
    "选择学习模块：",
    ["首页", "Transformer架构", "自注意力机制", "模型训练可视化", "实际应用案例"],
    key="page",
    on_change=on_page_change
)

# 应用页眉
st.title("PyTorch与Transformer可视化教学系统")

# 页面内容
if page == "首页":
    st.markdown("""
    ## 欢迎使用PyTorch与Transformer可视化教学系统

    本系统旨在通过交互式可视化帮助您理解Transformer架构和PyTorch框架。

    ### 系统提供的主要功能：
    - **Transformer架构**：直观了解Transformer的各个组件及其工作原理
    - **自注意力机制**：交互式演示Self-Attention的计算过程
    - **模型训练可视化**：观察神经网络训练过程中的参数变化
    - **实际应用案例**：体验Transformer在实际任务中的应用

    ### 使用指南：
    使用左侧的导航栏选择您想要学习的模块，每个模块都提供了交互式的学习体验。
    """)
    
    st.image(r"./6miQ.png",
             caption="Transformer架构示意图")

elif page == "Transformer架构":
    # 添加锚点，确保页面滚动到顶部
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    show_transformer_architecture()

elif page == "自注意力机制":
    # 添加锚点，确保页面滚动到顶部
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    visualize_attention()

elif page == "模型训练可视化":
    # 添加锚点，确保页面滚动到顶部
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    show_training_visualization()

elif page == "实际应用案例":
    # 添加锚点，确保页面滚动到顶部
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.header("实际应用案例")
    
    application = st.selectbox(
        "选择应用案例：",
        ["文本生成", "情感分析", "机器翻译"]
    )
    
    if application == "文本生成":
        st.subheader("文本生成示例")
        prompt = st.text_area("输入提示文本：", "今天天气真不错，我决定")
        if st.button("生成文本"):
            with st.spinner("正在生成..."):
                # 在实际应用中，这里应该调用文本生成模型
                generated_text = prompt + "去公园散步。在公园里，我看到了很多人在享受阳光，有的在打羽毛球，有的在跑步，还有的在野餐。这样的好天气确实让人心情愉悦。"
                st.success("生成完成！")
                st.write(generated_text)
    
    elif application == "情感分析":
        st.subheader("情感分析示例")
        text = st.text_area("输入待分析文本：", "这部电影太棒了，剧情引人入胜，演员表演也非常出色！")
        if st.button("分析情感"):
            with st.spinner("正在分析..."):
                # 在实际应用中，这里应该调用情感分析模型
                sentiment = "积极 (置信度: 0.92)"
                st.success(f"分析结果：{sentiment}")
    
    elif application == "机器翻译":
        st.subheader("机器翻译示例")
        source_lang = st.selectbox("源语言：", ["英语", "中文", "法语", "德语"])
        target_lang = st.selectbox("目标语言：", ["中文", "英语", "法语", "德语"])
        text = st.text_area("输入待翻译文本：", "Hello, world! How are you today?")
        if st.button("翻译"):
            with st.spinner("正在翻译..."):
                # 在实际应用中，这里应该调用翻译模型
                translated_text = "你好，世界！今天你好吗？"
                st.success("翻译完成！")
                st.write(translated_text)

# 页脚
st.markdown("---")
st.markdown("### 关于本系统")
st.markdown("本系统是一个用于学习PyTorch和Transformer的教学工具，旨在通过可视化和交互式演示帮助理解深度学习概念。") 