import cv2
import os
import platform
import sys


def detect_camera():
    print("Searching for camera named 'usb2.0'...")

    # Detect operating system
    os_name = platform.system().lower()
    print(f"Platform: {os_name}")

    # Get list of available cameras
    cap = None
    for i in range(10):  # Check first 10 possible camera indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.get(cv2.CAP_PROP_BACKEND)
                fourcc = cap.get(cv2.CAP_PROP_FOURCC)

                # Try to get camera name from properties
                camera_name = f"Camera {i}"
                try:
                    camera_name = cap.get(cv2.CAP_PROP_DEVICE_NAME)
                    if not camera_name:
                        camera_name = f"Camera {i}"
                except:
                    pass

                # Get device information based on OS
                device_info = "N/A"
                if os_name == "linux":
                    device_path = f"/dev/video{i}"
                    if os.path.exists(device_path):
                        try:
                            with open(
                                f"/sys/class/video4linux/video{i}/name", "r"
                            ) as f:
                                device_info = f.read().strip()
                        except:
                            device_info = "N/A"
                    else:
                        device_info = "N/A"
                elif os_name == "windows":
                    try:
                        # Try to get Windows device name
                        device_info = f"Windows Camera {i}"
                        # Additional Windows-specific code could go here
                    except:
                        device_info = "N/A"
                else:
                    device_info = "N/A"

                # print(f"\nFound camera: {camera_name}")
                # print(f"  Index: {i}")
                # print(f"  Resolution: {width}x{height}")
                # print(f"  FPS: {fps}")
                # print(f"  Backend: {backend}")
                # print(f"  FourCC: {fourcc}")
                # print(f"  Device Info: {device_info}")

                # Check if this is the camera we're looking for
                if "usb 2.0" in camera_name.lower() or "usb 2.0" in device_info.lower():
                    print("\n\u2705 Found target camera 'usb2.0'!")
                    cap.release()
                    return i

                cap.release()
        except:
            pass
    return None


if __name__ == "__main__":
    id = detect_camera()
    cap = cv2.VideoCapture(id)
    if not cap.isOpened():
        print("无法打开摄像头")
        cap.release()
        cv2.destroyAllWindows()

    else:
        while True:
            # 读取一帧画面
            ret, frame = cap.read()

            # 检查是否成功读取帧
            if not ret:
                print("错误：无法读取摄像头画面！")
                break

            cv2.imshow("demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # 按q退出
                print("用户主动退出测试")
                break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

