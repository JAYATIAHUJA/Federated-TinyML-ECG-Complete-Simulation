import subprocess
import sys
import time

def main():
    scripts = [
        "02_baseline_fl.py",
        "03_privacy_sweep.py",
        "04_lmic_dropout.py",
        "05_quantise.py",
        "06_generate_figures.py"
    ]

    print("=====================================================")
    print(">> Starting End-to-End FL Pipeline with Improved Models")
    print("=====================================================\n")

    start_time = time.time()

    for script in scripts:
        print(f"\n[{time.strftime('%H:%M:%S')}] Executing: {script}")
        print("-" * 50)
        
        result = subprocess.run([sys.executable, "-u", script])
        
        if result.returncode != 0:
            print(f"\n[ERROR] {script} failed with exit code {result.returncode}. Pipeline stopped.")
            sys.exit(1)
            
        print(f"[OK] {script} completed successfully.")

    total_time = (time.time() - start_time) / 60
    print(f"\n[SUCCESS] Pipeline completed successfully in {total_time:.1f} minutes!")
    print("All figures have been regenerated in the `figures/` directory.")

if __name__ == "__main__":
    main()
