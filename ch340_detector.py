#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH340 Serial Port Auto-Detection Script
自动搜索CH340串口设备并输出对应的COM口号
兼容Linux
"""

import serial.tools.list_ports


def detect_ch340_port():
    port_list = list(serial.tools.list_ports.comports())
    for port_info in port_list:
        port = port_info.device
        description = port_info.description
        hwid = port_info.hwid

        if "CH340" in description.upper() or "1a86:7523" in hwid.lower():
                print("=" * 50)
                print(f"{" " * 14}✓ 发现CH340设备")
                print(f"{" " * 16}端口号:{port}")
                print("=" * 50)
                return port
        
if __name__ == "__main__":
    detect_ch340_port()