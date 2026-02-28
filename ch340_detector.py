#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH340 Serial Port Auto-Detection Script
自动搜索CH340串口设备并输出对应的COM口号
"""

import serial
import serial.tools.list_ports


def detect_ch340_port():
    port_list = list(serial.tools.list_ports.comports())
    for com_port in port_list:
        port = com_port.device
        description = com_port.description

        if "CH340" in description.upper():
                print(f"{" " * 14}✓ 发现CH340设备")
                return port