"""Quick training status check. Run: python jepa_status.py"""
import torch, os, time, subprocess

RUNS = [
    ("lqt0.5", "/tmp/r_lqt0.5.pt"),
    ("lqt1.0", "/tmp/r_lqt1.0.pt"),
    ("lqt2.0", "/tmp/r_lqt2.0.pt"),
]

print("\nDownloading resume checkpoints...")
for label, path in RUNS:
    subprocess.run(
        ["modal", "volume", "get", "--force", "ptbxl-data",
         f"/jepa/jepa_linear_{label}_resume.pt", path],
        capture_output=True,
    )

now = time.time()
print(f"\n{'='*58}")
print(f"  JEPA Status  {time.strftime('%H:%M:%S %Z')}")
print(f"{'='*58}")
for label, path in RUNS:
    if not os.path.exists(path):
        print(f"  !! {label}  -- no checkpoint found")
        continue
    d = torch.load(path, map_location="cpu", weights_only=False)
    ep   = d["next_epoch"] - 1
    mae  = d["best_val_mae"]
    r    = d["best_val_r"]
    age  = now - os.path.getmtime(path)
    mins = int(age // 60)
    secs = int(age % 60)
    flag = "OK " if age < 300 else "!!!"
    mae_str = f"{mae:.2f} ms" if mae < 1e9 else "  --  "
    print(f"  [{flag}] {label}  FT {ep:>3}/150  MAE={mae_str}  R={r:.3f}  ({mins}m {secs}s ago)")
print(f"{'='*58}")
print("  [!!!] = no update in >5 min — likely crashed")
print(f"{'='*58}\n")
