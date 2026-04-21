import subprocess
import time
import threading
import sys
import os

# 配置区域
CPP_EXECUTABLE = (
    "/home/ironegg/thesis/robot/bin/robot_control"  # 根据 CMake 编译结果确定的路径
)


class RobotController:
    def __init__(self, executable_path):
        self.executable_path = executable_path
        self.process = None
        self.is_running = False
        self.read_thread = None

    def start(self):
        """启动 C++ 进程并建立通信"""
        if not os.path.exists(self.executable_path):
            print(
                f"错误: 找不到文件 {self.executable_path}。请先运行 cmake --build build"
            )
            return False

        try:
            # 启动子进程
            # stdin: 写入指令, stdout: 读取日志/里程计, stderr: 读取报错
            self.process = subprocess.Popen(
                [self.executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
            )
            self.is_running = True

            # 启动后台线程读取输出
            self.read_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.read_thread.start()

            # 启动后台线程监控错误
            self.error_thread = threading.Thread(target=self._error_loop, daemon=True)
            self.error_thread.start()

            print(f"已启动进程: {self.executable_path} (PID: {self.process.pid})")
            return True
        except Exception as e:
            print(f"启动失败: {e}")
            return False

    def _reader_loop(self):
        """处理 C++ 的标准输出 (stdout)"""
        try:
            for line in iter(self.process.stdout.readline, ""):
                clean_line = line.strip()
                if clean_line:
                    print(f"| [C++] {clean_line}", flush=True)

                    # 可以在这里根据关键字触发 Python 回调
                    if "所有动作序列执行完成！" in clean_line:
                        print("通知: 机器人已完成所有预定动作")
        except Exception as e:
            if self.is_running:
                print(f"读取线程异常: {e}")

    def _error_loop(self):
        """监听 C++ 的错误输出 (stderr)，捕获如'串口打开失败'等信息"""
        try:
            for line in iter(self.process.stderr.readline, ""):
                if line:
                    print(f"⚠️  [C++ Error] {line.strip()}", file=sys.stderr)
        except:
            pass

    def send_command(self, cmd):
        """向 C++ 发送指令"""
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(f"{cmd}\n")
                self.process.stdin.flush()
                print(f"-> [Python] 发送指令: {cmd}")
                return True
            except BrokenPipeError:
                print("❌ 错误: 管道已断开。C++ 进程可能已崩溃。")
                self.is_running = False
        else:
            print("❌ 错误: C++ 进程未运行。")
        return False

    def stop(self):
        """停止进程"""
        self.is_running = False
        if self.process:
            self.process.terminate()
            print("进程已终止")


def main():
    bot = RobotController(CPP_EXECUTABLE)

    if not bot.start():
        return

    try:
        # 给 C++ 一点初始化时间
        time.sleep(1)

        # 检查进程是否还在（可能因为串口权限直接挂了）
        if bot.process.poll() is not None:
            print("❌ C++ 进程在初始化阶段退出，请检查上方 Error 输出。")
            return

        print("\n--- 开始控制流程 ---")

        # 发送回车/START 指令触发 C++ 逻辑
        bot.send_command("START")

        # 模拟监控 10 秒
        count = 0
        while count < 10:
            time.sleep(1)
            count += 1
            if bot.process.poll() is not None:
                break

        # 发送停止指令
        bot.send_command("STOP")
        try:
            bot.process.wait(timeout=3.0)  # 给 C++ 3秒钟的时间完成刹车和清理
            print("C++ 进程已正常退出")
        except subprocess.TimeoutExpired:
            print("⚠️ C++ 进程未能在超时时间内退出，将强制终止")
            bot.stop()

    except KeyboardInterrupt:
        print("\n用户手动中断")
        bot.send_command("STOP")  # 同样发送 STOP 让其优雅退出
        try:
            bot.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            bot.stop()
    finally:
        if bot.process and bot.process.poll() is None:
            bot.stop()


if __name__ == "__main__":
    main()
