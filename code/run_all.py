from __future__ import annotations
import os, sys, subprocess

def run(cmd, cwd):
    print('\n$ ' + ' '.join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

if __name__ == '__main__':
    root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
    py=sys.executable
    run([py,'code/generate_data.py'], root)
    run([py,'code/train_detect.py'], root)
    run([py,'code/evaluate.py'], root)
    run([py,'code/plot_utils.py'], root)
    print('\nDone. Outputs are in data/, ground_truth/, plots/, and report.pdf')
