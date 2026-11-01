from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[1]
class CropArray:
 def __init__(self,key):self.key=key
 def __getitem__(self,row):return np.load(R/'assets/crops'/f'{self.key}_{int(row)}.npy')
class CropFile:
 def __init__(self,path,*args,**kwargs):self.key=Path(path).stem
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def __getitem__(self,key):
  assert key=='crops'
  return CropArray(self.key)
