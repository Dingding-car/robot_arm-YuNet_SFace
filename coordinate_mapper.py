import cv2
import numpy as np
import yaml
import sys
sys.path.append('./kinematic')
from kinematic.arm5dof_uservo import Arm5DoFUServo
from ch340_detector import detect_ch340_port
from camera_detector import detect_camera


class CoordinateMapper:
    def __init__(self, calibration_file, camera_int_file, camera_ext_file, obj_height = 0):
        # 标定板数据和相机参数路径
        self.calibration_file = calibration_file
        self.camera_int_file = camera_int_file
        self.camera_ext_file = camera_ext_file 

        # 标定板9点坐标
        self.points = []
        self.pixel_coords = []
        self.workspace_coords = []
        # 相机内参
        self.camera_matrix = []
        self.dist_coeffs = []
        # 相机外参（相机坐标系到工作台坐标系变换矩阵）
        self.T_cam2ws = []
        self.rvec_cam2ws = []
        self.tvect_cam2ws = []
        # 工件高度
        self.obj_height = obj_height
        # 返回的工作台坐标
        self.selected_workspace_coord = None

        self._load_calibration()

        self.update_affine_matrix()

    def _load_calibration(self):
        # 标定板参数
        with open(self.calibration_file, 'r') as file:
            data = yaml.safe_load(file)
            for point in data['calibration_points']:
                self.points.append(point)
                self.pixel_coords.append([point['pixel_coordinates']['u'], point['pixel_coordinates']['v']])
                self.workspace_coords.append([point['workspace_coordinates']['x'], point['workspace_coordinates']['y']])
        # 相机内参
        data = np.load(self.camera_int_file)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        # 相机外参
        self.T_cam2ws = np.loadtxt(self.camera_ext_file, delimiter=',')
        self.rvec_cam2ws = cv2.Rodrigues(self.T_cam2ws[:3, :3])[0]
        self.tvec_cam2ws = self.T_cam2ws[:3, 3]

    def _get_ws_subpoints(self):
        # 给9点赋值Z轴坐标
        workspace_coords_3d = np.zeros((9, 3))
        workspace_coords_3d[:, :2] = self.workspace_coords
        workspace_coords_3d[:, 2] = self.obj_height
        # print(type(workspace_coords_3d))

        # 空间点投影到图片上
        ws2img_9points, _ = cv2.projectPoints(
            workspace_coords_3d,
            self.rvec_cam2ws,
            self.tvec_cam2ws,
            self.camera_matrix,
            self.dist_coeffs
        )
        ws2img_9points = ws2img_9points.reshape((-1, 2))
        return ws2img_9points


    def update_affine_matrix(self):
        ws_subpoints = self._get_ws_subpoints()
        # print(f"ws_subpoints type: {type(ws_subpoints)}, shape: {ws_subpoints.shape}")
        # print(f"workspace_coords type: {type(self.workspace_coords)}, shape: {np.array(self.workspace_coords).shape}")
        self.affine2d_matrix = cv2.estimateAffine2D(ws_subpoints, np.array(self.workspace_coords))[0]
        return self.affine2d_matrix
    

    def pixel_to_workspace(self, u, v):
            # Using the linear mapping: [u, v] = [a*x + b*y + c, d*x + e*y + f]
            # We need to solve for [x, y] given [u, v]
            a, b, c, d, e, f = self.affine2d_matrix.reshape(-1)
            wx = a*u + b*v + c
            wy = d*u + e*v + f
            return (wx, wy)

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            workspace_coord = self.pixel_to_workspace(x, y)
            self.selected_workspace_coord = workspace_coord
            if workspace_coord is not None:
                print(f"Selected pixel: ({x}, {y}) -> Workspace: ({workspace_coord[0]:.2f}, {workspace_coord[1]:.2f})")
            else:
                print(f"Selected pixel: ({x}, {y}) -> Outside calibration area")

def main():

    calib_file = './config/calibration_points.yaml'
    cam_int_file = './config/camera_calib_params.npz'
    cam_ext_file = './config/T_cam2ws.txt'

    mapper = CoordinateMapper(calib_file, cam_int_file, cam_ext_file)

    SERVO_PORT = detect_ch340_port()
    CAMERA_ID = detect_camera()

    arm = Arm5DoFUServo(SERVO_PORT)
    print("机械臂运动到拍照点")
    arm.set_tool_pose(pose_name='capture_image')
    cap = cv2.VideoCapture(CAMERA_ID)

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', mapper.select_point)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw calibration points on the frame for reference
        for point in mapper.points:
            u = point['pixel_coordinates']['u']
            v = point['pixel_coordinates']['v']
            cv2.circle(frame, (int(u), int(v)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Point {point['point_id']}", (int(u)+10, int(v)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()