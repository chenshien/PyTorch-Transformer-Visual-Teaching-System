#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文字体安装脚本
用于自动下载并安装中文字体，以解决图表中中文显示为方块的问题
"""

import os
import sys
import platform
import subprocess
import time
import shutil

def print_title():
    """打印标题"""
    print("=" * 60)
    print("PyTorch与Transformer可视化教学系统 - 中文字体安装助手")
    print("=" * 60)
    print("此脚本用于解决图表中文字显示为方块的问题")
    print("=" * 60)

def check_chinese_fonts():
    """检查系统中的中文字体"""
    print("\n正在检查系统中的中文字体...")
    system = platform.system()
    
    if system == "Windows":
        # Windows系统
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        chinese_fonts = ["msyh.ttc", "simhei.ttf", "simsun.ttc"]
        
        for font in chinese_fonts:
            if os.path.exists(os.path.join(font_dir, font)):
                print(f"✓ 找到中文字体: {font}")
            else:
                print(f"✗ 未找到中文字体: {font}")
        
    elif system == "Darwin":  # macOS
        # macOS系统
        try:
            output = subprocess.check_output(['fc-list', ':lang=zh']).decode('utf-8')
            if output.strip():
                print("✓ 系统中找到中文字体")
                font_lines = output.strip().split('\n')
                for line in font_lines[:5]:  # 只显示前5个
                    print(f"  - {line.split(':')[0]}")
                if len(font_lines) > 5:
                    other_fonts_count = len(font_lines) - 5
                    print(f"  ... 以及 {other_fonts_count} 个其他中文字体")
            else:
                print("✗ 系统中未找到中文字体")
        except:
            print("✗ 无法检查中文字体，可能需要安装fontconfig")
    
    else:  # Linux
        # Linux系统
        try:
            output = subprocess.check_output(['fc-list', ':lang=zh']).decode('utf-8')
            if output.strip():
                print("✓ 系统中找到中文字体")
                font_lines = output.strip().split('\n')
                for line in font_lines[:5]:  # 只显示前5个
                    print(f"  - {line.split(':')[0]}")
                if len(font_lines) > 5:
                    other_fonts_count = len(font_lines) - 5
                    print(f"  ... 以及 {other_fonts_count} 个其他中文字体")
            else:
                print("✗ 系统中未找到中文字体")
        except:
            print("✗ 无法检查中文字体，可能需要安装fontconfig")

def download_font(url, save_path):
    """从网络下载字体"""
    print(f"\n正在从 {url} 下载字体...")
    
    try:
        # 尝试使用 urllib
        import urllib.request
        urllib.request.urlretrieve(url, save_path)
        if os.path.exists(save_path):
            print(f"✓ 字体已成功下载到: {save_path}")
            return True
        return False
    except:
        print("✗ 使用urllib下载失败，尝试其他方法...")
    
    try:
        # 尝试使用 requests
        import requests
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        if os.path.exists(save_path):
            print(f"✓ 字体已成功下载到: {save_path}")
            return True
        return False
    except:
        print("✗ 使用requests下载失败，尝试其他方法...")
    
    # 尝试使用系统命令
    try:
        if platform.system() == "Windows":
            # Windows使用PowerShell
            cmd = f'powershell -Command "Invoke-WebRequest -Uri {url} -OutFile {save_path}"'
            subprocess.call(cmd, shell=True)
        else:
            # Linux/macOS使用curl或wget
            if subprocess.call(['which', 'curl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                subprocess.call(['curl', '-L', url, '-o', save_path])
            elif subprocess.call(['which', 'wget'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                subprocess.call(['wget', url, '-O', save_path])
            else:
                print("✗ 系统中未找到curl或wget，无法下载")
                return False
        
        if os.path.exists(save_path):
            print(f"✓ 字体已成功下载到: {save_path}")
            return True
        else:
            print("✗ 下载失败")
            return False
    except Exception as e:
        print(f"✗ 下载过程出错: {e}")
        return False

def install_font(font_path):
    """安装字体到系统"""
    print(f"\n正在安装字体: {font_path}...")
    
    if not os.path.exists(font_path):
        print(f"✗ 字体文件不存在: {font_path}")
        return False
    
    system = platform.system()
    
    if system == "Windows":
        # Windows系统安装字体
        try:
            # 复制到用户字体目录
            user_font_dir = os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts")
            if not os.path.exists(user_font_dir):
                os.makedirs(user_font_dir)
            
            font_name = os.path.basename(font_path)
            dest_path = os.path.join(user_font_dir, font_name)
            shutil.copy2(font_path, dest_path)
            
            # 注册字体
            import ctypes
            from ctypes import wintypes
            gdi32 = ctypes.WinDLL('gdi32')
            FONTS_REG_PATH = r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts'
            
            user_choice = input("是否安装字体到系统? (y/n，默认为n): ").strip().lower()
            if user_choice == 'y':
                # 尝试使用管理员权限安装到系统
                try:
                    system_font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
                    sys_dest_path = os.path.join(system_font_dir, font_name)
                    shutil.copy2(font_path, sys_dest_path)
                    print(f"✓ 字体已安装到系统: {sys_dest_path}")
                except:
                    print("✗ 无法安装到系统字体目录，已安装到用户字体目录")
            
            print(f"✓ 字体已成功安装到: {dest_path}")
            print("注意: 可能需要重新启动应用程序以使更改生效")
            return True
        except Exception as e:
            print(f"✗ 安装字体时出错: {e}")
            return False
    
    elif system == "Darwin":  # macOS
        # macOS系统安装字体
        try:
            user_font_dir = os.path.expanduser("~/Library/Fonts")
            if not os.path.exists(user_font_dir):
                os.makedirs(user_font_dir)
            
            font_name = os.path.basename(font_path)
            dest_path = os.path.join(user_font_dir, font_name)
            shutil.copy2(font_path, dest_path)
            
            # 清理字体缓存
            user_choice = input("是否刷新字体缓存? (y/n，默认为y): ").strip().lower()
            if user_choice != 'n':
                try:
                    subprocess.call(['atsutil', 'databases', '-remove'])
                    print("✓ 字体缓存已刷新")
                except:
                    print("✗ 无法刷新字体缓存")
            
            print(f"✓ 字体已成功安装到: {dest_path}")
            print("注意: 可能需要重新启动应用程序以使更改生效")
            return True
        except Exception as e:
            print(f"✗ 安装字体时出错: {e}")
            return False
    
    else:  # Linux
        # Linux系统安装字体
        try:
            user_font_dir = os.path.expanduser("~/.local/share/fonts")
            if not os.path.exists(user_font_dir):
                os.makedirs(user_font_dir)
            
            font_name = os.path.basename(font_path)
            dest_path = os.path.join(user_font_dir, font_name)
            shutil.copy2(font_path, dest_path)
            
            # 更新字体缓存
            user_choice = input("是否更新字体缓存? (y/n，默认为y): ").strip().lower()
            if user_choice != 'n':
                try:
                    subprocess.call(['fc-cache', '-f', '-v'])
                    print("✓ 字体缓存已更新")
                except:
                    print("✗ 无法更新字体缓存")
            
            print(f"✓ 字体已成功安装到: {dest_path}")
            print("注意: 可能需要重新启动应用程序以使更改生效")
            return True
        except Exception as e:
            print(f"✗ 安装字体时出错: {e}")
            return False

def manual_installation_guide():
    """显示手动安装指南"""
    system = platform.system()
    print("\n===== 手动安装指南 =====")
    
    if system == "Windows":
        print("""
Windows系统手动安装中文字体:
1. 下载中文字体，如微软雅黑(Microsoft YaHei)、宋体(SimSun)或黑体(SimHei)
2. 右键点击下载的字体文件，选择"安装"或"为所有用户安装"
3. 重新启动应用程序
""")
    elif system == "Darwin":  # macOS
        print("""
macOS系统手动安装中文字体:
1. 下载中文字体，如苹果丽黑(Heiti SC)或思源黑体(Source Han Sans)
   可以使用Homebrew: brew cask install font-noto-sans-cjk
2. 也可以双击.ttf/.otf文件使用字体册安装
3. 重新启动应用程序
""")
    else:  # Linux
        print("""
Linux系统手动安装中文字体:
1. 对于Ubuntu/Debian：
   sudo apt-get update
   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

2. 对于Fedora：
   sudo dnf install wqy-microhei-fonts wqy-zenhei-fonts

3. 对于Arch Linux：
   sudo pacman -S wqy-microhei wqy-zenhei

4. 手动安装:
   a. 下载字体文件(.ttf或.otf)
   b. 创建目录: mkdir -p ~/.local/share/fonts
   c. 复制字体: cp 字体文件.ttf ~/.local/share/fonts/
   d. 更新缓存: fc-cache -f -v

5. 重新启动应用程序
""")

def main():
    """主函数"""
    print_title()
    
    # 检查中文字体
    check_chinese_fonts()
    
    # 询问用户是否要下载并安装字体
    user_choice = input("\n是否要下载并安装中文字体? (y/n): ").strip().lower()
    
    if user_choice != 'y':
        print("\n您选择不下载字体。如果需要手动安装中文字体，请参考以下指南：")
        manual_installation_guide()
        return
    
    # 准备下载字体
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_save_path = os.path.join(script_dir, "chinese_font.ttf")
    
    # 字体下载URLs
    font_urls = [
        "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    ]
    
    # 尝试下载字体
    download_success = False
    for url in font_urls:
        if download_font(url, font_save_path):
            download_success = True
            break
    
    if not download_success:
        print("\n✗ 无法自动下载字体文件")
        print("请尝试手动安装中文字体：")
        manual_installation_guide()
        return
    
    # 安装字体
    if install_font(font_save_path):
        print("\n✓ 字体安装过程已完成")
    else:
        print("\n✗ 字体安装过程失败")
        print("请尝试手动安装中文字体：")
        manual_installation_guide()
    
    # 完成
    print("\n" + "=" * 60)
    print("字体安装程序已完成。请重新启动应用程序以应用更改。")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消。")
    except Exception as e:
        print(f"\n\n程序出错: {e}")
        print("请尝试手动安装中文字体。")
        manual_installation_guide()
    
    input("\n按回车键退出...") 