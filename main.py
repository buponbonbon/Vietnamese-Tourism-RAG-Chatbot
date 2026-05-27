import os
import subprocess
import sys

def main_control_center():
    """
    Central Controller Gateway.
    Provides programmatic shortcuts to trigger localized environments without manual CLI inputs.
    """
    print("\n" + "="*60)
    print("   VIETNAMESE TOURISM RAG - SYSTEM MANAGEMENT CONTROL CENTER   ")
    print("="*60)
    print("[1] Launch Local Production Server (Flask Backend + CUDA Acceleration)")
    print("[2] Launch Cloud Simulation Engine (Streamlit Interface + CPU Fallback)")
    print("[3] Exit Control Gateway")
    print("="*60)

    choice = input("Select execution routine [1-3]: ").strip()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if choice == "1":
        # Routes control directly to activate the Flask app/local_backend.py
        target_path = os.path.join(current_dir, "app", "local_backend.py")
        print(f"\n[BOOT] Initializing Flask Production instances via: {target_path}\n")
        try:
            subprocess.run([sys.executable, target_path], check=True)
        except KeyboardInterrupt:
            print("\n[INFO] Local Flask server terminated safely.")

    elif choice == "2":
        # Routes control directly to simulate the Streamlit app.py in CPU mode
        target_path = os.path.join(current_dir, "app.py")
        print(f"\n[BOOT] Initializing Streamlit Cloud simulations via: {target_path}\n")
        try:
            subprocess.run(["streamlit", "run", target_path], check=True)
        except KeyboardInterrupt:
            print("\n[INFO] Streamlit local simulation terminated safely.")

    elif choice == "3":
        print("\n[INFO] Exiting management control center. Goodbye!")
        sys.exit(0)
    else:
        print("\n[ERROR] Invalid parameter token selection. System aborting.")

if __name__ == "__main__":
    main_control_center()