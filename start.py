#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyTorch与Transformer可视化教学系统启动脚本
这个脚本用于启动可视化教学系统，并确保中文显示正常
"""

import os
import platform
import subprocess
import sys
import time
import webbrowser
import shutil

def check_dependencies():
    """检查必要的依赖是否已安装"""
    try:
        import streamlit
        import torch
        import matplotlib
        import numpy
        import pandas
        import seaborn
        return True
    except ImportError as e:
        print(f"缺少必要的依赖: {e}")
        print("请先运行: pip install -r requirements.txt")
        return False

def check_chinese_fonts():
    """检查系统中是否安装了中文字体"""
    system = platform.system()
    has_chinese_font = False
    
    if system == "Windows":
        # Windows系统
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        chinese_fonts = ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
        
        for font in chinese_fonts:
            if os.path.exists(os.path.join(font_dir, font)):
                has_chinese_font = True
                print(f"找到中文字体: {font}")
                break
    
    elif system == "Darwin":  # macOS
        # macOS系统
        try:
            output = subprocess.check_output(['fc-list', ':lang=zh'], stderr=subprocess.PIPE).decode('utf-8')
            if output.strip():
                has_chinese_font = True
                print("系统中找到中文字体")
        except:
            try:
                # 检查常见字体目录
                font_dirs = [
                    "/Library/Fonts",
                    "/System/Library/Fonts",
                    os.path.expanduser("~/Library/Fonts")
                ]
                chinese_font_patterns = [
                    "*Hei*.ttf", "*Hei*.ttc", "*PingFang*.ttc", "*Song*.ttf", "*Song*.ttc",
                    "*Noto*CJK*", "*Source*Han*"
                ]
                
                for directory in font_dirs:
                    if not os.path.exists(directory):
                        continue
                    
                    for pattern in chinese_font_patterns:
                        import glob
                        matches = glob.glob(os.path.join(directory, pattern))
                        if matches:
                            has_chinese_font = True
                            print(f"找到中文字体: {os.path.basename(matches[0])}")
                            break
                    
                    if has_chinese_font:
                        break
            except:
                pass
    
    else:  # Linux
        # Linux系统
        try:
            output = subprocess.check_output(['fc-list', ':lang=zh'], stderr=subprocess.PIPE).decode('utf-8')
            if output.strip():
                has_chinese_font = True
                print("系统中找到中文字体")
        except:
            # 检查常见字体包
            font_packages = {
                "Debian/Ubuntu": ["fonts-wqy-microhei", "fonts-wqy-zenhei"],
                "Fedora": ["wqy-microhei-fonts", "wqy-zenhei-fonts"],
                "Arch": ["wqy-microhei", "wqy-zenhei"]
            }
            
            # 尝试手动检查一些常见路径
            font_paths = [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
                "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
                os.path.expanduser("~/.fonts/wqy-microhei.ttc"),
                os.path.expanduser("~/.local/share/fonts/wqy-microhei.ttc")
            ]
            
            for path in font_paths:
                if os.path.exists(path):
                    has_chinese_font = True
                    print(f"找到中文字体: {os.path.basename(path)}")
                    break
    
    return has_chinese_font

def install_missing_fonts():
    """尝试安装缺少的中文字体（如果需要）"""
    # 检查是否有中文字体
    has_chinese_font = check_chinese_fonts()
    
    if has_chinese_font:
        print("系统中已经安装了中文字体，无需额外安装。")
        return True
    
    print("\n系统中未检测到中文字体，图表中的中文可能会显示为方块。")
    
    # 如果存在中文字体安装脚本，建议用户运行
    if os.path.exists("install_chinese_font.py"):
        print("\n检测到字体安装脚本，建议运行以解决中文显示问题：")
        print("  python install_chinese_font.py")
        
        user_choice = input("\n是否现在运行字体安装脚本? (y/n): ").strip().lower()
        if user_choice == 'y':
            try:
                subprocess.call([sys.executable, "install_chinese_font.py"])
                print("\n字体安装完成，继续启动应用程序...")
                return True
            except:
                print("\n无法运行字体安装脚本，请手动运行：")
                print("  python install_chinese_font.py")
    else:
        system = platform.system()
        print("\n请安装中文字体：")
        
        if system == "Windows":
            print("Windows系统: 请安装微软雅黑(Microsoft YaHei)、宋体(SimSun)或黑体(SimHei)")
        
        elif system == "Darwin":  # macOS
            print("""macOS系统: 请安装中文字体:
    brew tap homebrew/cask-fonts
    brew install --cask font-noto-sans-cjk""")
        
        else:  # Linux
            print("""Linux系统:
    Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
    Fedora: sudo dnf install wqy-microhei-fonts wqy-zenhei-fonts
    Arch Linux: sudo pacman -S wqy-microhei wqy-zenhei""")
    
    print("\n更多详细信息请参见README.md中的中文字体安装说明。")
    
    user_choice = input("\n是否继续启动应用程序? (y/n): ").strip().lower()
    if user_choice != 'y':
        print("启动已取消，请安装中文字体后再次运行。")
        return False
    
    return True

def generate_example_data():
    """生成示例数据文件"""
    try:
        from utils.demo_data import save_example_data
        save_example_data()
        print("已生成示例数据。")
    except Exception as e:
        print(f"生成示例数据时出错: {e}")

def start_application():
    """启动应用程序"""
    try:
        print("正在启动PyTorch与Transformer可视化教学系统...")
        
        # 检查是否为PyCharm或其他IDE环境
        in_ide = "PYCHARM_HOSTED" in os.environ or "VSCODE_PID" in os.environ
        
        if in_ide:
            # 在IDE中以模块方式运行
            print("在IDE中运行应用...")
            os.system(f"{sys.executable} -m streamlit run app.py")
        else:
            # 直接运行应用并打开浏览器
            print("启动应用程序服务器...")
            
            # 使用subprocess.Popen运行streamlit
            streamlit_process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py", "--no-watchdog"],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待Streamlit启动（查找"You can now view your Streamlit app"消息）
            started = False
            port = 8501  # 默认Streamlit端口
            for line in iter(streamlit_process.stdout.readline, ""):
                print(line, end="")
                if "You can now view your Streamlit app in your browser" in line:
                    started = True
                if "localhost:" in line and started:
                    # 尝试提取端口
                    try:
                        port = int(line.split("localhost:")[1].split()[0])
                    except:
                        pass
                    break
                if "Address already in use" in line:
                    # 如果端口被占用，尝试查找新端口
                    try:
                        port_part = line.split("localhost:")[1]
                        port = int(''.join(filter(str.isdigit, port_part)))
                    except:
                        pass
            
            # 启动浏览器
            if started:
                time.sleep(1)  # 给服务器一点启动时间
                print(f"在浏览器中打开应用: http://localhost:{port}")
                webbrowser.open(f"http://localhost:{port}")
                
                # 监听输出直到用户中断
                try:
                    for line in iter(streamlit_process.stdout.readline, ""):
                        print(line, end="")
                except KeyboardInterrupt:
                    print("\n用户中断，关闭应用程序...")
                    streamlit_process.terminate()
            else:
                print("应用程序未能成功启动，请查看错误信息。")
                # 输出任何错误
                for line in iter(streamlit_process.stderr.readline, ""):
                    print(line, end="")
                    if line == "":
                        break
    except Exception as e:
        print(f"启动应用程序时出错: {e}")

def main():
    """主函数"""
    print("=" * 50)
    print("PyTorch与Transformer可视化教学系统")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("缺少必要的依赖，请安装后再次运行。")
        return
    
    # 检查/安装字体
    if not install_missing_fonts():
        return
    
    # 生成示例数据
    generate_example_data()
    
    # 启动应用
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已被用户中断。")
    except Exception as e:
        print(f"\n程序出错: {e}")
        
    # 在Windows环境中，保持控制台窗口打开
    if platform.system() == "Windows" and "PYCHARM_HOSTED" not in os.environ and "VSCODE_PID" not in os.environ:
        input("\n按回车键退出...") 