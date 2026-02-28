import cv2
import numpy as np
import threading
import queue
import time

import sys
sys.path.append('./kinematic')

from kinematic.arm5dof_uservo import Arm5DoFUServo
from model.yunet import YuNet
from model.sface import SFace
from model.PIDController import PIDController2D
from ch340_detector import detect_ch340_port



# 人脸检测可视化函数
def visualize(image, results, matches=None, scores=None, box_color=(0, 255, 0), text_color=(255, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    # 处理空值，避免索引错误
    if matches is None:
        matches = []
    if scores is None:
        scores = []

    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for idx, det in enumerate(results):
        # 动态设置框颜色：匹配目标人脸=绿色，未匹配=红色
        current_box_color = box_color
        if idx < len(matches):
            current_box_color = (0, 255, 0) if matches[idx] else (0, 0, 255)

        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), current_box_color, 2)

        # 检测置信度
        conf = det[-1]
        cv2.putText(output, 'DET:{:.2f}'.format(conf), (bbox[0]+2, bbox[1]+14), cv2.FONT_HERSHEY_DUPLEX, 0.5, current_box_color)

        # 显示匹配分数和匹配状态（仅当有匹配结果时）
        if idx < len(scores) and idx < len(matches):
            # 匹配分数
            score_text = 'MAT: {:.2f}'.format(scores[idx])
            cv2.putText(output, score_text, (bbox[0]+2, bbox[1]+30), cv2.FONT_HERSHEY_DUPLEX, 0.5, current_box_color)
            # 匹配状态
            match_text = 'Matched' if matches[idx] else 'Not matched'
            cv2.putText(output, match_text, (bbox[0], bbox[1]-8), cv2.FONT_HERSHEY_DUPLEX, 0.5, current_box_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv2.circle(output, landmark, 2, landmark_color[idx], 2)

    return output

def init_YuNet(model_path):
    # YuNet模型初始化
    model = YuNet(modelPath=model_path,
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
    )
    return model

def init_SFace(model_path):
    # SFace模型初始化
    model = SFace(model_path)
    return model

# 检测结果队列（用于主线程和舵机控制线程之间的通信）
servo_queue = queue.Queue(maxsize=1)

def capture_video(target_path, camera_id = 0):

    # YuNet模型初始化
    YN_model_path = './model/face_detection_yunet_2023mar.onnx'
    detector = init_YuNet(YN_model_path)

    # SFace模型初始化
    SF_model_path = './model/face_recognition_sface_2021dec.onnx'
    recognizer = init_SFace(SF_model_path)

    # 打开摄像头
    deviceId = camera_id # 摄像头设备ID,默认为0
    cap = cv2.VideoCapture(deviceId)

    # 检查视频是否成功打开
    global stop_servo_thread
    if not cap.isOpened():
        print("Error: Could not open video.")
        stop_servo_thread = True
        return
    
    # 检测目标人脸
    target = cv2.imread(target_path)
    detector.setInputSize([target.shape[1], target.shape[0]])
    target_face = detector.infer(target)
    assert target_face.shape[0] > 0, 'Cannot find a face in target'

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([w, h])

    tm = cv2.TickMeter()

    print('Press "q" to quit the demo.')
    while cv2.waitKey(1) != ord('q'):
        tm.start()

        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference and match
        scores = []
        matches = []
        detected_faces = detector.infer(frame) # results is a tuple shape[N,15]
        for face in detected_faces:
            recognized_face = recognizer.match(target, target_face[0][:-1], frame, face[:-1])
            scores.append(recognized_face[0])
            matches.append(recognized_face[1])
        
        # 向舵机队列放入检测结果（非阻塞，避免阻塞主线程）
        try:
            # 放入结果、画面宽度、高度
            servo_queue.put_nowait((detected_faces, w, h))
        except queue.Full:
            # 队列满时移除旧数据，放入新数据
            servo_queue.get_nowait()
            servo_queue.put_nowait((detected_faces, w, h))

        tm.stop()
        frame = visualize(frame, detected_faces, matches=matches, scores=scores ,fps=tm.getFPS())

        # Visualize results in a new Window
        cv2.imshow('Track and check', frame)

        tm.reset()
    
    # 释放资源（必须执行，否则会导致摄像头被占用）
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
    
    # 通知舵机线程停止
    stop_servo_thread = True

def servo_control(servo_manager, stop_servo_thread= False):

    # PID参数设置
    Kp = (1e-2, 5e-3)
    Ki = (1e-3, 1e-3)
    Kd = (0, 0)

    # 初始化二维PID控制器
    # 可针对X/Y设置不同参数：比如水平响应快一点，垂直稳一点
    pid_2d = PIDController2D(
        kp=Kp,
        ki=Ki,
        kd=Kd,
        min_output=(-90, -90),  # X/Y输出下限
        max_output=(90, 90),    # X/Y输出上限
        integral_limit=10       # 积分限幅
    )

    servo_raw_angle = servo_manager.get_servo_angle_list()
    pan_angle = servo_raw_angle[0]  # 初始化云台水平角度
    tilt_angle = servo_raw_angle[2] # 初始化云台垂直角度

    while not stop_servo_thread:
        try:
            # 非阻塞获取队列数据
            detected_faces, dispW, dispH = servo_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.01)
            continue

        if servo_manager is None or len(detected_faces) <= 0:
            pid_2d.reset()
            continue
        
        
        face = detected_faces[0]
        # 鼻尖坐标
        nose = (face[8], face[9])   # (nose_x, nose_y)

        # 设定点
        setpoint = (dispW // 2, dispH // 2)

        # PID输出
        pid_output = pid_2d.compute(setpoint_2d= setpoint, feedback_2d= nose)

        pan_angle -= pid_output[0]
        tilt_angle += pid_output[1]

        # 最终角度限位（双重保险）
        pan_angle = np.clip(pan_angle, -90, 90)
        tilt_angle = np.clip(tilt_angle, -90, 90)


        # 执行舵机控制
        try:
            servo_manager.uservo.set_servo_angle(0, pan_angle)
            servo_manager.uservo.set_servo_angle(2, tilt_angle)
            # print(f"二维PID输出 | 水平角度: {pan_angle:.1f}° | 垂直角度: {tilt_angle:.1f}° | "
            #       f"反馈点: {nose}")
        except Exception as e:
            print(f"舵机控制异常: {e}")
            continue


# 主函数
def main():
    # //TODO 舵机串口号
    print("=" * 50)
    SERVO_PORT = detect_ch340_port()
    print(f"{" " * 16}端口号:{SERVO_PORT}")
    print("=" * 50)
    
    servo_manager = Arm5DoFUServo(device=SERVO_PORT, is_init_pose= False)
    print("机械臂回正")
    servo_manager.home()

    # 启动舵机控制线程
    servo_thread = threading.Thread(target=servo_control, args=(servo_manager,), daemon=True)
    servo_thread.start()
    print("舵机控制线程已启动")

    capture_video(target_path='./images/target3.jpg',camera_id=0)

    # 等待舵机线程结束
    servo_thread.join(timeout=1)
    print("舵机正常退出")

    
    # 舵机归位
    servo_manager.home()
    servo_manager.set_damping(1200)



if __name__ == "__main__":
    main()