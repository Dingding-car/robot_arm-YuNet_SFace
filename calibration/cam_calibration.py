import cv2
import numpy as np
import glob
import os

# ===================== 核心参数配置（只需修改这里）=====================
# 1. 标定图像相关
IMG_DIR = "./calib_image"       # 标定图像存放目录
IMG_EXT = "jpg"                # 标定图像格式（jpg/png）
# 2. 棋盘格参数
CHESSBOARD_COLS = 9            # 棋盘格内角点列数（横向）
CHESSBOARD_ROWS = 6            # 棋盘格内角点行数（纵向）
SQUARE_SIZE = 19.0             # 每个棋盘格方块的实际尺寸（单位：mm）
# 3. 结果保存
SAVE_PATH = "./config/camera_calib_params.npz"  # 标定结果保存路径
# 4. 验证选项
VALIDATE = False                # 是否显示畸变校正效果（True/False）

def calibrate_camera():
    """相机标定主函数"""
    # ===================== 1. 初始化参数 =====================
    # 棋盘格内角点数量
    chessboard_size = (CHESSBOARD_COLS, CHESSBOARD_ROWS)
    # 亚像素角点优化的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 生成3D世界坐标（棋盘格平面z=0）
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # 缩放至实际物理尺寸

    # 存储所有图像的3D点和2D点
    obj_points = []  # 3D世界点
    img_points = []   # 2D图像点
    img_shape = None  # 图像尺寸（所有图像需一致）

    # ===================== 2. 读取并处理标定图像 =====================
    # 获取所有标定图像路径
    image_paths = glob.glob(os.path.join(IMG_DIR, f"*.{IMG_EXT}"))
    if not image_paths:
        print(f"❌ 错误：在 {IMG_DIR} 目录下未找到 {IMG_EXT} 格式的图像！")
        print("   请检查目录是否存在，或修改 IMG_DIR/IMG_EXT 参数")
        return

    print(f"📁 找到 {len(image_paths)} 张标定图像，开始检测角点...")
    print("-" * 50)
    
    for idx, img_path in enumerate(image_paths):
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  警告：跳过无法读取的图像 → {os.path.basename(img_path)}")
            continue
        
        # 检查图像尺寸是否一致
        current_shape = img.shape[:2]
        if img_shape is None:
            img_shape = current_shape
        elif current_shape != img_shape:
            print(f"⚠️  警告：图像尺寸不一致 → {os.path.basename(img_path)}，跳过")
            continue
        
        # 转为灰度图（角点检测需要灰度图）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格内角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # 亚像素级角点优化（提升标定精度）
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 保存3D-2D点对
            obj_points.append(objp)
            img_points.append(corners2)
            
            # 绘制角点并可视化（可选）
            img_draw = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow(f"角点检测 ({idx+1}/{len(image_paths)})", img_draw)
            cv2.waitKey(200)  # 每张图显示200ms
            print(f"✅ 成功 → {os.path.basename(img_path)}")
        else:
            print(f"❌ 失败 → {os.path.basename(img_path)}（角点检测不到）")

    cv2.destroyAllWindows()
    print("-" * 50)

    # 检查有效标定图像数量（至少需要10张才能保证精度）
    if len(obj_points) < 10:
        print(f"❌ 错误：有效标定图像仅 {len(obj_points)} 张（至少需要10张）")
        print("   请补充更多不同角度的棋盘格照片，或检查照片质量")
        return

    # ===================== 3. 执行相机标定 =====================
    print("\n🔧 开始相机标定计算...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )

    # ===================== 4. 评估标定精度 =====================
    # 计算平均重投影误差（越小越好，<1为优秀，<3为可用）
    mean_error = 0
    for i in range(len(obj_points)):
        img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        mean_error += error
    mean_error /= len(obj_points)

    # ===================== 5. 输出并保存标定结果 =====================
    print("\n" + "="*50)
    print("📷 相机标定结果")
    print("="*50)
    print(f"📊 标定重投影误差（整体）：{ret:.4f}")
    print(f"📊 平均重投影误差：{mean_error:.4f} (越小越好，<1为佳)")
    print("\n📐 相机内参矩阵 (cameraMatrix)：")
    print(np.round(camera_matrix, 4))  # 保留4位小数，更易读
    print("\n🔍 畸变系数 (distCoeffs) [k1, k2, p1, p2, k3]：")
    print(np.round(dist_coeffs, 6))    # 保留6位小数

    # 保存标定结果到文件
    np.savez(
        SAVE_PATH,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        img_shape=img_shape,
        mean_error=mean_error
    )
    print(f"\n💾 标定结果已保存至：{os.path.abspath(SAVE_PATH)}")

    # ===================== 6. 验证：畸变校正效果 =====================
    if VALIDATE and len(image_paths) > 0:
        # 读取第一张图像做校正演示
        test_img = cv2.imread(image_paths[0])
        h, w = test_img.shape[:2]
        
        # 获取最优校正后的相机矩阵（去除无用黑边）
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h)
        )
        
        # 执行畸变校正
        dst = cv2.undistort(test_img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        
        # 裁剪图像（去除校正后的黑边）
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
        
        # 显示校正前后对比
        cv2.imshow("Original Image (原始图像)", test_img)
        cv2.imshow("Undistorted Image (校正后)", dst)
        print("\n📸 显示畸变校正效果，按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存校正示例图像
        cv2.imwrite("undistorted_example.jpg", dst)
        print("✅ 校正示例图像已保存为：undistorted_example.jpg")

if __name__ == "__main__":
    # 执行相机标定
    calibrate_camera()