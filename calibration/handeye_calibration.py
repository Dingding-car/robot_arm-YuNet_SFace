import cv2
import numpy as np
import yaml
import sys
sys.path.append('./kinematic')

from kinematic.arm5dof_uservo import Arm5DoFUServo
from ch340_detector import detect_ch340_port
from camera_detector import detect_camera

# 全局变量存储点击的点
global clicked_points
clicked_points = []

class Workspace_calibration:
    def __init__(self, camera_params_file, board_width, board_height):
        self.camera_matrix, self.dist_coeffs = self._load_camera_params(camera_params_file)
        self.board_width = board_width
        self.board_height = board_height

    def _load_camera_params(self, npz_file):
        """
        加载相机内参（npz格式）
        """
        data = np.load(npz_file)
        camera_matrix = data['camera_matrix']
        dist_coeff = data['dist_coeffs']
        return camera_matrix, dist_coeff

    def _generate_ws_points(self):
        """
        生成工作台坐标系下九个点的三维坐标（Z=0）
        顺序：P0 (x0, y0), P1 (x0, 0), P2 (x0, -y0),
            P3 (0, y0),  P4 (0, 0),  P5 (0, -y0),
            P6 (-x0, y0), P7 (-x0, 0), P8 (-x0, -y0)
        其中 x0 = board_height/2, y0 = board_width/2
        """
        x0 = self.board_height / 2.0
        y0 = self.board_width / 2.0
        ws_9points = np.array([
            [x0,  y0, 0],  # P0
            [x0,  0,  0],  # P1
            [x0, -y0, 0],  # P2
            [0,   y0, 0],  # P3
            [0,   0,  0],  # P4
            [0,  -y0, 0],  # P5
            [-x0, y0, 0],  # P6
            [-x0, 0,  0],  # P7
            [-x0,-y0, 0]   # P8
        ], dtype=np.float32)
        return ws_9points
    
    def solve_T_cam2ws(self, img_9points):
        """
        求解相机坐标系到工作台坐标系的变换矩阵
        :param img_9points: 9个点的像素坐标，np.array (9,2) float32
        :return: rvec, tvec 旋转向量和平移向量
        """
        ws_9points = self._generate_ws_points()
        # 使用PnP求解位姿
        success, rvec, tvec = cv2.solvePnP(
            ws_9points, 
            img_9points, 
            self.camera_matrix, 
            self.dist_coeffs
        )
        if success:
            # 构造相机坐标系到工作台坐标系的变换矩阵
            T_cam2ws = np.eye(4)
            T_cam2ws[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cam2ws[:3, 3] = tvec.reshape(-1)
            print("成功求解相机到工作台的变换矩阵")
            print("旋转向量:\n", rvec)
            print("平移向量:\n", tvec)
            return success, rvec, tvec, ws_9points
        else:
            print("PnP求解失败")
            return None, None

def mouse_callback(event, x, y, flags, param):
    """
    鼠标回调函数：双击左键记录像素坐标
    :param event: 鼠标事件
    :param x: 鼠标x坐标
    :param y: 鼠标y坐标
    :param flags: 鼠标标志
    :param param: 传入的图像对象
    """
    global clicked_points
    image = param  # 获取传入的图像
    
    # 双击左键且未收集满9个点时记录坐标
    if event == cv2.EVENT_LBUTTONDBLCLK and len(clicked_points) < 9:
        print(f"已选点 {len(clicked_points)+1}: ({x}, {y})")
        clicked_points.append([x, y])
        
        # 绘制实心圆标记点
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        # 绘制点的序号
        cv2.putText(image, str(len(clicked_points)), (x+10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

def get_9points_from_camera(camera_id = 0):
    """
    从摄像头获取画面，通过鼠标双击获取9个点的像素坐标
    :return: 9个点的像素坐标 np.array (9,2) float32
    """
    global clicked_points
    clicked_points = []  # 重置点列表
    
    # 打开摄像头（0为默认摄像头）
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception("无法打开摄像头，请检查设备连接")
    
    # 创建窗口并设置鼠标回调
    cv2.namedWindow("Workspace Calibration - Double click to select 9 points", cv2.WINDOW_NORMAL)
    
    print("请在画面中双击左键选择9个点，按Q键退出，选满9个点自动结束")
    
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        # 复制帧用于绘制（避免修改原帧）
        display_frame = frame.copy()
        
        # 绘制已选择的点
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (x+10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 显示提示文字
        cv2.putText(display_frame, f"Selected points: {len(clicked_points)}/9", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 设置鼠标回调（传入当前帧用于绘制）
        cv2.setMouseCallback("Workspace Calibration - Double click to select 9 points", 
                            mouse_callback, display_frame)
        
        # 显示画面
        cv2.imshow("Workspace Calibration - Double click to select 9 points", display_frame)
        
        # 退出条件：ESC键 或 选满9个点
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # ESC键
            print("用户手动退出")
            cap.release()
            cv2.destroyAllWindows()
            return None
        if len(clicked_points) >= 9:
            print("已选满9个点，完成采集")
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    
    # 转换为numpy数组并返回
    return np.array(clicked_points, dtype=np.float32)

def save_calibration_points_to_yaml(ws_points, img_points, yaml_path):
    """
    将实际坐标（工作台坐标系）和像素坐标保存为YAML文件
    :param ws_points: 9个点的实际三维坐标，np.array (9,3)
    :param img_points: 9个点的像素坐标，np.array (9,2)
    :param yaml_path: YAML文件保存路径
    """
    # 构造数据字典
    calibration_data = {
        "calibration_points": [
            {
                "point_id": i+1,
                "workspace_coordinates": {  # 实际坐标（工作台坐标系）
                    "x": float(ws_points[i][0]),
                    "y": float(ws_points[i][1]),
                    "z": float(ws_points[i][2])
                },
                "pixel_coordinates": {  # 像素坐标
                    "u": float(img_points[i][0]),
                    "v": float(img_points[i][1])
                }
            } for i in range(9)
        ]
    }
    
    # 写入YAML文件
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"标定点数据已保存到YAML文件: {yaml_path}")

def handeye_calib():
    # 1. 配置参数
    camera_params_file = "./config/camera_calib_params.npz"  # 相机内参文件路径
    board_width = 105.0  # 工作台宽度（单位：mm）
    board_height = 79.0  # 工作台高度（单位：mm）
    yaml_save_path = "./config/calibration_points.yaml" # YAML文件保存路径
    
    # 自动检测串口和相机
    SERVO_PORT = detect_ch340_port()
    CAMERA_ID = detect_camera()

    servo_manager = Arm5DoFUServo(device=SERVO_PORT, is_init_pose= False)
    print("机械臂运动到拍照点")
    servo_manager.set_tool_pose(pose_name='capture_image', T=2.0)

    # 2. 获取9个点的像素坐标
    print("开始从摄像头采集9个点...")
    img_9points = get_9points_from_camera(CAMERA_ID)
    if img_9points is None or len(img_9points) != 9:
        servo_manager.set_damping(1000)
        print("点采集失败或未完成")
        return
    
    print("采集到的9个点像素坐标：")
    for i, (x, y) in enumerate(img_9points):
        print(f"P{i}: ({x:.2f}, {y:.2f})")
    
    # 3. 初始化标定类并求解变换矩阵
    try:
        calib = Workspace_calibration(camera_params_file, board_width, board_height)
        success, rvec, tvec, ws_9points = calib.solve_T_cam2ws(img_9points)
        
        # 如果需要，可以将旋转向量转换为旋转矩阵
        if success:
            # 构造相机坐标系到工作台坐标系的变换矩阵
            T_cam2ws = np.eye(4)
            T_cam2ws[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cam2ws[:3, 3] = tvec.reshape(-1)
            print("变换矩阵:\n", T_cam2ws)

            # 保存变换矩阵和标定点的像素坐标和工作台坐标
            np.savetxt('./config/T_cam2ws.txt', T_cam2ws, fmt='%.3f', delimiter=',')
            save_calibration_points_to_yaml(ws_9points, img_9points, yaml_save_path)

    except FileNotFoundError as e:
        print(f"路径 {e} 未找到，请检查路径")
    except Exception as e:
        print(f"标定过程出错: {e}")
    
    servo_manager.set_damping(1000)
