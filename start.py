import subprocess
import os
def main():
     # Start the Streamlit application
    print("Starting Streamlit application...")
    local_dir = os.path.dirname(os.path.abspath(__file__))
    process = subprocess.Popen(["streamlit", "run", os.path.join(local_dir, "ui", "Welcome.py"), "--server.maxUploadSize", "1024"],)

if __name__ == "__main__":
    main()
    print("Streamlit application started successfully.")
   