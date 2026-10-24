"""Create a small synthetic fixture; these are not experimental results."""
from pathlib import Path
import numpy as np

out = Path("runs/synthetic.npz")
out.parent.mkdir(exist_ok=True)
if out.exists():
    raise SystemExit(f"Refusing to overwrite {out}")
np.savez(out, query=np.array([[1.,0.],[0.,1.],[1.,1.]]),
         gallery=np.array([[1.,0.],[0.,1.],[-1.,0.]]),
         positive=np.array([[True,False,False],[False,True,False],[False,False,False]]))
print(out)
