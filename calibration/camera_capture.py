#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
import time
from datetime import datetime

def capture_photo():
    """
    使用OpenCV调用摄像头拍照并保存
    """
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return False
    
    print("摄像头已打开，准备拍照...")
    print("按空格键拍照，按ESC键退出")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 显示实时画面
        cv2.imshow('Camera', frame)
        
        # 等待按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC键退出
            break
        elif key == 32:  # 空格键拍照
            # 生成文件名（使用时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            save_directory = "calib_images" # 指定保存目录
            if not os.path.exists(save_directory):
                os.makedirs(save_directory) # 如果目录不存在，则创建
            filename = f"caliboard_{timestamp}.jpg"
            save_path = os.path.join(save_directory, filename) # 保存文件路径
            
            # 保存照片
            cv2.imwrite(save_path, frame)
            print(f"照片已保存为: {save_path}")
            
            # 显示拍照成功提示
            cv2.putText(frame, "Photo Taken!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
            cv2.waitKey(500)  # 显示拍照提示0.5秒
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    return True

if __name__ == "__main__":
    capture_photo()