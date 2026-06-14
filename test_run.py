
import subprocess
import sys

try:
    process = subprocess.Popen(
        [r".\venv\Scripts\python.exe", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=r"."
    )
    stdout, stderr = process.communicate(timeout=60)
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)
except subprocess.TimeoutExpired:
    process.kill()
    stdout, stderr = process.communicate()
    print("STDOUT (Timeout):")
    print(stdout)
    print("STDERR (Timeout):")
    print(stderr)
except Exception as e:
    print(f"Error: {e}")
