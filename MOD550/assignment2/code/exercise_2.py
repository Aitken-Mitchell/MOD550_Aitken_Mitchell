import subprocess
import sys

def run_script(script_name):
    """Run a Python script using subprocess."""
    try:
        print(f"Running {script_name}...")
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"{script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        sys.exit(1)

def main():
    # List of scripts to run in order
    scripts = ["step1.py", "step2.py", "step3.py", "step4.py", "step5.py", "step6.py", "step7.py", "step8.py"]
    

    # Run each script sequentially
    for script in scripts:
        run_script(script)

    print("All steps completed successfully.")

if __name__ == "__main__":
    main()