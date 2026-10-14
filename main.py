import os
import subprocess
import sys
import traceback


def run_command(command, target_path):
    if not os.path.isfile(target_path):
        print(f"\n[ERROR] Không tìm thấy file:")
        print(target_path)
        return

    try:
        subprocess.run(command, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))

    except KeyboardInterrupt:
        print("\n[INFO] Tiến trình đã được dừng.")

    except subprocess.CalledProcessError as error:
        print(f"\n[ERROR] Tiến trình kết thúc với mã lỗi: {error.returncode}")

    except Exception as error:
        print(f"\n[ERROR] {type(error).__name__}: {error}")
        traceback.print_exc()


def main_control_center():
    print("\n" + "=" * 60)
    print("VIETNAMESE TOURISM RAG - CONTROL CENTER")
    print("=" * 60)
    print("[1] Chạy Flask Backend (127.0.0.1:5000)")
    print("[2] Chạy Streamlit (127.0.0.1:8501)")
    print("[3] Thoát")

    choice = input("Chọn [1-3]: ").strip()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if choice == "1":
        target_path = os.path.join(
            current_dir,
            "app",
            "local_backend.py"
        )

        print("Đang chạy:", target_path)

        run_command(
            [sys.executable, target_path],
            target_path
        )

    elif choice == "2":
        target_path = os.path.join(current_dir, "app.py")

        print("Đang chạy:", target_path)

        run_command(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                target_path
            ],
            target_path
        )

    elif choice == "3":
        print("Đã thoát.")
        sys.exit(0)

    else:
        print("[ERROR] Chỉ được nhập 1, 2 hoặc 3.")


if __name__ == "__main__":
    main_control_center()