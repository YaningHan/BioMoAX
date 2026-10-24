"""Read-only checks of the processed training input contract."""
from pathlib import Path
import h5py
import numpy as np


CELLS = {"u2os":690,"a549":1039,"a375":86,"mcf7":85,"ht29":78}
H5_FILES = [
    "crops/crops_cpg0012_690.h5", "crops/crops_cpg0004_a549x1055.h5",
    "crops/crops_cpg0016_A375.h5", "crops/crops_cpg0016_MCF7.h5",
    "crops/crops_cpg0016_HT29.h5", "crops_cpg0030_u2os.h5",
] + [f"retraining_candidate_v1/cpg0000_{kind}/crops_cpg0000_{'cpd_' if kind=='compound' else ''}{cell}.h5"
     for kind in ["crispr","compound"] for cell in ["A549","U2OS"]]
METADATA = [
    "metadata/L1000_JumpCP_meta.csv", "metadata/l1000_a549_1039_meta.csv",
    "metadata/cpg0016_A375_meta.csv", "metadata/cpg0016_MCF7_meta.csv",
    "metadata/cpg0016_HT29_meta.csv", "metadata/cpg0012_profiles_merged.csv",
    "metadata/compoundinfo_beta.txt", "metadata/cpg0030_train_split.csv",
    "metadata/crispr_gene_split_v8_90_40_seed42.csv",
    "metadata/u2os_cpg0012_train473_site_split_seed42.csv",
    "metadata/SingleCell_Manifest_JumpCP_690.csv", "meta_cpg0030_u2os.csv",
    "retraining_candidate_v1/complete.json",
] + [f"{prefix}cpg0000_{kind}/meta_cpg0000_{'cpd_' if kind=='compound' else ''}{cell}.csv"
     for prefix in ["","retraining_candidate_v1/"]
     for kind in ["crispr","compound"] for cell in ["A549","U2OS"]]


def check_inputs(project_root):
    root=Path(project_root)/"data/processed"
    features={f"features/l1000_{c}_{n}.npy":n for c,n in CELLS.items()}
    required=[*H5_FILES,*METADATA,*features]
    report={"root":str(root),"missing":[],"invalid":[],"checked_files":0}
    for name in required:
        path=root/name
        if not path.is_file():
            report["missing"].append(name);continue
        report["checked_files"]+=1
        try:
            if name in H5_FILES:
                with h5py.File(path,"r") as h:
                    crops=h["crops"]
                    if crops.ndim!=4 or tuple(crops.shape[1:])!=(5,128,128) or crops.dtype!=np.uint16:
                        raise ValueError("Expected uint16 crops [N,5,128,128]")
                    # cpg0030 stores labels in row-aligned external CSV metadata.
                    if name != "crops_cpg0030_u2os.h5":
                        labels=h["labels"]
                        if labels.shape!=(len(crops),) or not np.issubdtype(labels.dtype,np.integer):
                            raise ValueError("Expected one integer label per crop")
            elif name in features:
                x=np.load(path,mmap_mode='r',allow_pickle=False)
                if x.shape!=(features[name],12328):
                    raise ValueError(f"Expected {(features[name],12328)}, got {x.shape}")
                if not np.isfinite(x).all():raise ValueError("Nonfinite expression values")
        except (OSError,ValueError,KeyError) as e:
            report["invalid"].append({"path":name,"reason":str(e)})
    report["passed"]=not report["missing"] and not report["invalid"]
    return report
