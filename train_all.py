"""Run the original base and incremental stages sequentially with matched settings."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description="Base -> incremental training")
p.add_argument("--dataset", default="Houston")
p.add_argument("--data_path", default="/mnt/HDD/data/ZY/EHSnet-net/TGRS_2026_LEAF/datasets/")
p.add_argument("--save_path", default="./results/")
p.add_argument("--cuda", type=int, default=0)
p.add_argument("--num_epoch", type=int, default=100)
p.add_argument("--num_epoch_inc", type=int, default=500)
p.add_argument("--repeat_time", type=int, default=1)
p.add_argument("--training_sample_ratio", type=float, default=0.8)
p.add_argument("--re_ratio", type=int, default=5)
p.add_argument("--gpt_path", default=None)
p.add_argument("--alpha", type=float, default=0.2)
p.add_argument("--alpha2", type=float, default=0.8)
p.add_argument("--a", type=float, default=0)
p.add_argument("--beta", type=float, default=0)
p.add_argument("--gama", type=float, default=0)
p.add_argument("--lr", type=float, default=0.05)
p.add_argument("--patch_size", type=int, default=13)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--weightalpha", type=float, default=0.01)
p.add_argument("--ensemble_alpha", type=float, default=0.1)
p.add_argument("--depth", type=int, default=1)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--skip_base", action="store_true", help="Use existing base checkpoints")
args = p.parse_args()
shared = ['dataset','data_path','save_path','cuda','num_epoch','repeat_time','training_sample_ratio','re_ratio','alpha','alpha2','a','beta','gama','lr','patch_size','batch_size']
def cmd(script, keys):
    out = [sys.executable, str(ROOT / script)]
    for k in keys:
        v = getattr(args, k)
        if v is not None: out += ['--'+k, str(v)]
    if args.gpt_path: out += ['--gpt_path', args.gpt_path]
    return out
if not args.skip_base:
    print('=== BASE TRAINING ===', flush=True)
    subprocess.run(cmd('train.py', shared), cwd=ROOT, check=True)
print('=== INCREMENTAL TRAINING ===', flush=True)
subprocess.run(cmd('train_incremental_proto.py', shared + ['num_epoch_inc','weightalpha','ensemble_alpha','depth','heads']), cwd=ROOT, check=True)
print('=== BOTH STAGES FINISHED ===', flush=True)
