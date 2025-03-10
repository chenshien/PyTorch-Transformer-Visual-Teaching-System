#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
紧急中文字体下载工具
此脚本用于下载一个应急的中文字体，确保图表中的中文可以正常显示
"""

import os
import sys
import urllib.request
import subprocess
import platform

def main():
    print("=" * 60)
    print("紧急中文字体下载工具")
    print("=" * 60)
    print("此工具将下载一个开源中文字体，确保图表中的中文可以正常显示。")
    
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, "emergency_font.ttf")
    
    # 检查是否已存在
    if os.path.exists(save_path):
        print(f"中文字体文件已存在: {save_path}")
        return
    
    # 尝试下载 Google Noto Sans CJK
    font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf"
    backup_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
    
    print(f"正在下载中文字体...")
    success = False
    
    # 尝试使用 urllib
    try:
        urllib.request.urlretrieve(font_url, save_path)
        if os.path.exists(save_path):
            print(f"字体下载成功: {save_path}")
            success = True
    except Exception as e:
        print(f"使用urllib下载失败: {e}")
    
    # 如果urllib失败，尝试使用系统命令
    if not success:
        try:
            if platform.system() == "Windows":
                # Windows使用PowerShell
                cmd = f'powershell -Command "Invoke-WebRequest -Uri {font_url} -OutFile {save_path}"'
                subprocess.call(cmd, shell=True)
            else:
                # Linux/macOS使用curl或wget
                if subprocess.call(['which', 'curl'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                    subprocess.call(['curl', '-L', font_url, '-o', save_path])
                elif subprocess.call(['which', 'wget'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) == 0:
                    subprocess.call(['wget', font_url, '-O', save_path])
                else:
                    print("找不到curl或wget，无法下载")
            
            if os.path.exists(save_path):
                print(f"字体下载成功: {save_path}")
                success = True
        except Exception as e:
            print(f"使用系统命令下载失败: {e}")
    
    # 如果仍然失败，尝试备用URL
    if not success:
        try:
            print("尝试使用备用链接下载...")
            urllib.request.urlretrieve(backup_url, save_path)
            if os.path.exists(save_path):
                print(f"使用备用链接下载成功: {save_path}")
                success = True
        except Exception as e:
            print(f"使用备用链接下载失败: {e}")
    
    if success:
        print("中文字体已成功下载，图表中的中文应该可以正常显示了。")
    else:
        print("所有下载尝试均失败，请手动下载中文字体。")
        print("您可以从以下链接下载中文字体:")
        print("1. Google Noto Sans CJK SC: https://www.google.com/get/noto/help/cjk/")
        print("2. 文泉驿字体: http://wenq.org/wqy2/index.cgi?Download")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序出错: {e}")
        
    input("按回车键退出...") 