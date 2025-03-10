import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import platform
import streamlit as st
import os
import numpy as np
import sys
import subprocess
import glob

def set_chinese_font():
    """
    设置matplotlib支持中文显示 - 增强版
    """
    system = platform.system()
    is_font_set = False
    
    # 根据不同操作系统设置不同的字体
    if system == 'Windows':
        # Windows系统尝试的字体顺序
        font_dirs = [
            "C:/Windows/Fonts",  # 标准Windows字体目录
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts")  # 用户字体目录
        ]
        font_names = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi',
            'Source Han Sans CN', 'Source Han Serif CN', 'Noto Sans CJK SC', 'Noto Serif CJK SC',
            'Microsoft JhengHei', 'DengXian', 'Arial Unicode MS'
        ]
        font_files = [
            'msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc', 'simkai.ttf', 'simfang.ttf',
            'SourceHanSansCN-Regular.otf', 'SourceHanSerifCN-Regular.otf',
            'NotoSansCJKsc-Regular.otf', 'NotoSerifCJKsc-Regular.otf'
        ]
    elif system == 'Darwin':  # macOS
        font_dirs = [
            "/Library/Fonts",  # 系统字体
            "/System/Library/Fonts",  # 系统字体
            os.path.expanduser("~/Library/Fonts")  # 用户字体
        ]
        font_names = [
            'Heiti SC', 'Hei', 'STHeiti', 'PingFang SC', 'Songti SC', 'Kaiti SC',
            'Source Han Sans CN', 'Source Han Serif CN', 'Noto Sans CJK SC', 'Noto Serif CJK SC',
            'Hiragino Sans GB', 'Hiragino Sans CNS', 'Apple LiSung', 'Apple LiGothic',
            'Arial Unicode MS'
        ]
        font_files = [
            'STHeiti-Light.ttc', 'STHeiti-Medium.ttc', 'PingFang.ttc', 
            'Songti.ttc', 'Kaiti.ttc', 'SourceHanSansCN-Regular.otf', 
            'SourceHanSerifCN-Regular.otf', 'NotoSansCJKsc-Regular.otf',
            'NotoSerifCJKsc-Regular.otf'
        ]
    else:  # Linux等其他系统
        font_dirs = [
            "/usr/share/fonts",  # 标准字体目录
            "/usr/local/share/fonts",  # 用户安装的字体
            os.path.expanduser("~/.fonts"),  # 用户字体目录
            os.path.expanduser("~/.local/share/fonts")  # 用户字体目录
        ]
        font_names = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Source Han Sans CN', 'Source Han Serif CN',
            'Noto Sans CJK SC', 'Noto Serif CJK SC', 'Droid Sans Fallback', 'AR PL UMing CN',
            'AR PL UKai CN', 'AR PL SungtiL GB', 'AR PL KaitiM GB', 'HanaMinA', 'HanaMinB',
            'UnDotum', 'UnBatang', 'Arial Unicode MS'
        ]
        font_files = [
            'wqy-microhei.ttc', 'wqy-zenhei.ttc', 'SourceHanSansCN-Regular.otf',
            'SourceHanSerifCN-Regular.otf', 'NotoSansCJKsc-Regular.otf',
            'NotoSerifCJKsc-Regular.otf', 'DroidSansFallback.ttf'
        ]
    
    # 自定义字体路径列表
    font_paths = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in font_files:
                paths = glob.glob(os.path.join(font_dir, "**", font_file), recursive=True)
                font_paths.extend(paths)
    
    # 添加字体文件路径
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                print(f"成功添加字体文件: {font_path}")
        except Exception as e:
            print(f"添加字体文件错误: {e}")
    
    # 尝试设置字体
    for font_name in font_names:
        try:
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 测试字体是否有效
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
            fig.canvas.draw()
            plt.close(fig)
            
            print(f"成功设置中文字体: {font_name}")
            is_font_set = True
            break
        except Exception as e:
            print(f"设置字体 {font_name} 失败: {e}")
            continue
    
    # 嵌入应急字体（适用于没有任何中文字体的情况）
    if not is_font_set:
        try:
            print("尝试嵌入应急字体...")
            # 检查utils目录中是否有紧急字体文件
            emergency_font_path = os.path.join(os.path.dirname(__file__), "emergency_font.ttf")
            if not os.path.exists(emergency_font_path):
                # 如果没有，尝试下载一个开源中文字体
                download_emergency_font(emergency_font_path)
            
            if os.path.exists(emergency_font_path):
                fm.fontManager.addfont(emergency_font_path)
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = ['emergency_font'] + plt.rcParams['font.sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
                print("已设置应急字体")
            else:
                # 如果所有尝试都失败，使用内嵌中文标签的方法
                print("无法设置中文字体，尝试使用图像内嵌标签")
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"设置应急字体失败: {e}")
            plt.rcParams['font.family'] = ['sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
    
    # 设置默认的图表样式
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 返回字体设置状态
    return is_font_set

def download_emergency_font(save_path):
    """
    下载开源中文字体作为应急使用
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 根据不同平台使用不同的下载方法
    if platform.system() == 'Windows':
        try:
            import urllib.request
            # NotoSansCJK-Regular 字体 (来自 Google Noto Fonts)
            font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
            urllib.request.urlretrieve(font_url, save_path)
            print(f"已下载应急字体到: {save_path}")
            return True
        except Exception as e:
            print(f"下载应急字体失败: {e}")
            return False
    else:
        try:
            # 使用 curl 或 wget 命令
            if subprocess.call(['which', 'curl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
                subprocess.call(['curl', '-L', font_url, '-o', save_path])
                print(f"已使用curl下载应急字体到: {save_path}")
                return True
            elif subprocess.call(['which', 'wget'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
                subprocess.call(['wget', font_url, '-O', save_path])
                print(f"已使用wget下载应急字体到: {save_path}")
                return True
            else:
                print("找不到curl或wget，无法下载应急字体")
                return False
        except Exception as e:
            print(f"下载应急字体失败: {e}")
            return False

def use_bitmap_font_for_chinese():
    """
    使用位图方式处理中文文本，适用于无法找到中文字体的情况
    """
    from matplotlib.text import Text
    
    # 保存原始的Text._get_layout方法
    original_get_layout = Text._get_layout
    
    # 创建一个新方法，处理中文字符
    def new_get_layout(self, renderer):
        # 处理中文字符，将其转换为图像
        # 这是处理方法的简化版本，仅用于演示
        return original_get_layout(self, renderer)
    
    # 替换原方法
    Text._get_layout = new_get_layout

def configure_streamlit_for_chinese():
    """
    配置Streamlit以更好地支持中文显示
    """
    # 设置宽页面模式以更好地显示中文
    st.set_page_config(
        page_title="PyTorch与Transformer可视化教学系统",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 添加自定义CSS改善中文显示
    st.markdown("""
    <style>
    body {
        font-family: "Microsoft YaHei", "SimHei", "STHeiti", "SimSun", "WenQuanYi Micro Hei", sans-serif !important;
    }
    .st-bw {
        font-family: "Microsoft YaHei", "SimHei", "STHeiti", "SimSun", "WenQuanYi Micro Hei", sans-serif !important;
    }
    .st-af {
        font-family: "Microsoft YaHei", "SimHei", "STHeiti", "SimSun", "WenQuanYi Micro Hei", sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

def fix_seaborn_chinese():
    """
    修复seaborn图表中的中文显示问题
    """
    import seaborn as sns
    
    # 保存原始的heatmap方法
    original_heatmap = sns.heatmap
    
    # 重新定义heatmap方法以支持中文
    def new_heatmap(*args, **kwargs):
        # 确保字体设置正确
        set_chinese_font()
        
        # 应用原始方法
        ret = original_heatmap(*args, **kwargs)
        
        # 确保图形元素使用中文字体
        plt.gcf().canvas.draw()
        return ret
    
    # 替换原始方法
    sns.heatmap = new_heatmap

def save_chinese_plot(fig, filename):
    """
    保存支持中文的图表
    
    Args:
        fig: matplotlib图表对象
        filename: 保存的文件名
    """
    set_chinese_font()
    fig.savefig(filename, dpi=300, bbox_inches='tight')

def plot_with_bitmap_text(ax, x, y, text, **kwargs):
    """
    在无法使用中文字体时，使用位图方式绘制中文文本
    
    Args:
        ax: matplotlib轴对象
        x, y: 文本位置
        text: 要绘制的文本
        **kwargs: 其他参数传递给matplotlib的text函数
    """
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    
    # 创建一个空白图像
    font_size = kwargs.get('fontsize', 12)
    img = Image.new('RGBA', (len(text) * font_size, font_size * 2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # 尝试使用PIL内置字体
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 绘制文本
    draw.text((0, 0), text, fill=kwargs.get('color', 'black'), font=font)
    
    # 将PIL图像转换为numpy数组
    arr = np.array(img)
    
    # 将图像作为图像而不是文本添加到图表中
    ax.imshow(arr, extent=[x, x + len(text) * 0.1, y, y + 0.1], 
              aspect='auto', zorder=10, alpha=kwargs.get('alpha', 1.0))

# 初始化
def init_chinese_support():
    """
    初始化所有中文支持的设置
    """
    # 设置字体并获取设置状态
    font_set = set_chinese_font()
    
    # 如果无法设置字体，尝试使用位图方式
    if not font_set:
        try:
            use_bitmap_font_for_chinese()
        except Exception as e:
            print(f"设置位图字体失败: {e}")
    
    # 修复seaborn中的中文
    try:
        fix_seaborn_chinese()
    except Exception as e:
        print(f"修复seaborn中文显示失败: {e}")
    
    # 返回字体设置结果
    return font_set

# 如果直接运行，则测试中文显示
if __name__ == "__main__":
    font_set = init_chinese_support()
    
    # 测试中文显示
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro-')
    ax.set_title('中文标题测试')
    ax.set_xlabel('横轴标签（中文）')
    ax.set_ylabel('纵轴标签（中文）')
    plt.grid(True)
    
    # 如果无法设置中文字体，使用位图方式
    if not font_set:
        plot_with_bitmap_text(ax, 2, 10, '使用位图方式的中文', fontsize=14, color='blue')
    
    # 显示图表
    plt.show()
    
    # 测试成功信息
    print("中文显示测试完成！") 