import sys
import subprocess
import time

modules = [
    ("Module 1: Satellite Image Analysis", "test_module1.py"),
    ("Module 2: Dataset Preparation & Tiling", "test_module2.py"),
    ("Module 3: Image Preprocessing", "test_module3.py"),
    ("Module 4: Traditional Change Detection", "test_module4.py"),
    ("Module 5: CNN Patch Classification", "test_module5.py"),
    ("Module 6: Semantic Segmentation (U-Net)", "test_module6.py"),
    ("Module 7: Siamese Network Feature Differencing", "test_module7.py"),
    ("Module 8: Advanced Attention Siamese U-Net & Benchmark", "test_module8.py"),
    ("Module 9: CNN + Transformer Hybrid", "test_module9.py"),
    ("Module 10: Model Training", "test_module10.py"),
]

print("=" * 80)
print("   SATELLITE IMAGE CHANGE DETECTION (SICD) — ALL 10 MODULES VERIFICATION")
print("=" * 80)

python_exe = sys.executable
all_success = True
timing_results = []

for idx, (mod_name, script) in enumerate(modules, 1):
    print(f"\n[{idx}/10] RUNNING {mod_name.upper()} ({script})...")
    start_t = time.time()
    
    proc = subprocess.run([python_exe, script], capture_output=True, text=True)
    elapsed = time.time() - start_t
    
    print(proc.stdout)
    if proc.stderr:
        print(f"STDERR:\n{proc.stderr}")
        
    if proc.returncode == 0:
        timing_results.append((mod_name, f"{elapsed:.2f}s", "PASSED [✓]"))
    else:
        timing_results.append((mod_name, f"{elapsed:.2f}s", "FAILED [✗]"))
        all_success = False

print("\n" + "=" * 80)
print("                     ALL 10 MODULES EXECUTION SUMMARY")
print("=" * 80)
print(f"{'Module Name':<55} | {'Duration':<10} | {'Status'}")
print("-" * 80)
for name, dur, status in timing_results:
    print(f"{name:<55} | {dur:<10} | {status}")
print("=" * 80)

if all_success:
    print("\n>>> [SUCCESS] ALL 10 MODULES TESTED, VERIFIED, AND FULLY FUNCTIONAL! <<<\n")
else:
    print("\n>>> [WARNING] One or more modules encountered errors. Please inspect the log above. <<<\n")
