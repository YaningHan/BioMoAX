"""Small command-line entry points; run `biomoax --help`."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser(description="BioMoAX model, retrieval and paper training")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("check-data", help="Check processed training file presence and array shapes")
    d.add_argument("--project-root", type=Path, required=True)
    e = sub.add_parser("evaluate", help="Evaluate prepared query/gallery embeddings")
    e.add_argument("--input", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--k", type=int, nargs="+", default=[1,5,10])
    r = sub.add_parser("ridge-fit", help="Fit both directions using training pairs only")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    en = sub.add_parser("encode", help="Encode crops of one condition, or expression rows")
    en.add_argument("--checkpoint", type=Path, required=True)
    en.add_argument("--input", type=Path, required=True)
    en.add_argument("--output", type=Path, required=True)
    en.add_argument("--modality", choices=["image","expression"], required=True)
    en.add_argument("--head", choices=["moa","compound"], default="moa")
    en.add_argument("--device", default="cpu")
    en.add_argument("--batch-size", type=int, default=16)
    t = sub.add_parser("train", help="Run the archived five-loss paper training recipe")
    t.add_argument("--project-root", type=Path, required=True, help="Root containing data/processed")
    t.add_argument("--output", type=Path, required=True, help="New training output directory")
    t.add_argument("--seed", type=int, default=43)
    t.add_argument("--preflight", action="store_true", help="Two updates with reduced caches")
    args = p.parse_args()
    if args.command == "check-data":
        from .data import check_inputs
        result=check_inputs(args.project_root)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result["passed"] else 1)
    if args.command == "train":
        env = os.environ.copy()
        env["BIOMOAX_PROJECT_ROOT"] = str(args.project_root.resolve())
        env["BIOMOAX_TRAIN_OUT"] = str(args.output.resolve())
        command = [sys.executable, str(Path(__file__).parent/"legacy/train_rebuilt_five_loss.py"), "--seed", str(args.seed)]
        if args.preflight: command.append("--preflight")
        raise SystemExit(subprocess.call(command, env=env))
    if args.output.exists():
        p.error(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "evaluate":
        from .metrics import retrieve
        with np.load(args.input, allow_pickle=False) as d:
            result = retrieve(d["query"], d["gallery"], d["positive"],
                              d["excluded"] if "excluded" in d else None, tuple(args.k))
        args.output.write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
    elif args.command == "ridge-fit":
        import joblib
        from .baselines import CrossModalRidge
        with np.load(args.input, allow_pickle=False) as d:
            model = CrossModalRidge().fit(d["image"], d["expression"])
        joblib.dump(model, args.output)
    elif args.command == "encode":
        from .inference import load_models, encode_condition
        models = load_models(args.checkpoint, args.device, args.head)
        z = encode_condition(models, np.load(args.input,allow_pickle=False),
                             args.modality,args.device,args.batch_size)
        with args.output.open("wb") as handle: np.save(handle,z)


if __name__ == "__main__":
    main()
