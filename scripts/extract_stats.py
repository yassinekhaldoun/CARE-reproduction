"""Kaggle side: predict the held-out patients of one fold and save per-patient statistics.

For every patient of the fold's validation list the script
  1. reads the 4 MRI modalities and the ground-truth labels,
  2. predicts softmax probabilities with the trained nnU-Net (never written to disk),
  3. compresses them into sufficient statistics (care.extract.case_stats),
and finally saves one small .npz plus a .json log of the settings.

Example (smoke test, 5 patients, mirroring off):
  python scripts/extract_stats.py \
      --model-dir  $nnUNet_results/Dataset001_BrainTumour/nnUNetTrainer_5epochs__nnUNetPlans__3d_fullres \
      --dataset-dir $nnUNet_raw/Dataset001_BrainTumour \
      --splits-file $nnUNet_preprocessed/Dataset001_BrainTumour/splits_final.json \
      --fold 0 --n-cases 5 --out /kaggle/working/stats_smoke.npz

Use --tta to turn mirroring test-time augmentation on (slower). Use --resume to continue a run that
was interrupted: patients already in --out are skipped.
"""
import argparse
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from care import extract, stats as S  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="nnU-Net results folder: <trainer>__<plans>__<configuration>")
    p.add_argument("--dataset-dir", required=True, help="nnUNet_raw/Dataset001_BrainTumour (imagesTr, labelsTr, dataset.json)")
    p.add_argument("--splits-file", required=True, help="splits_final.json of the preprocessed dataset")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--n-cases", type=int, default=None, help="only the first N held-out patients (smoke test)")
    p.add_argument("--tta", action="store_true", help="mirroring test-time augmentation (slower, nnU-Net default)")
    p.add_argument("--resume", action="store_true", help="skip patients already present in --out")
    p.add_argument("--out", required=True)
    return p.parse_args()


def rows_from_stats(stats):
    return [{k: v[i] for k, v in stats.items()} for i in range(S.n_cases(stats))]


def main():
    args = parse_args()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    dataset_dir, out = Path(args.dataset_dir), Path(args.out)
    n_channels = len(json.loads((dataset_dir / "dataset.json").read_text())["channel_names"])
    cases = json.loads(Path(args.splits_file).read_text())[args.fold]["val"]
    if args.n_cases:
        cases = cases[: args.n_cases]

    rows = []
    if args.resume and out.exists():
        rows = rows_from_stats(S.load_stats(out))
        print(f"resuming: {len(rows)} patients already done")
    done = {str(r["case_id"]) for r in rows}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=args.tta,
                                perform_everything_on_device=(device.type == "cuda"), device=device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.model_dir, use_folds=(args.fold,),
                                                   checkpoint_name=args.checkpoint)
    reader = SimpleITKIO()

    try:
        nnunet_version = version("nnunetv2")
    except PackageNotFoundError:
        nnunet_version = "unknown"
    seconds = []
    for k, case in enumerate(cases, 1):
        if case in done:
            continue
        t0 = time.time()
        image, props = reader.read_images([str(dataset_dir / "imagesTr" / f"{case}_{c:04d}.nii.gz")
                                           for c in range(n_channels)])
        seg, _ = reader.read_seg(str(dataset_dir / "labelsTr" / f"{case}.nii.gz"))
        _, probs = predictor.predict_single_npy_array(image, props, None, None, True)
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        st = extract.case_stats(probs, seg)
        st["case_id"] = np.array(case)
        rows.append(st)
        seconds.append(time.time() - t0)
        print(f"[{k}/{len(cases)}] {case}: voxels={int(st['n'])} tumour={int(st['syB'])} "
              f"true r={st['syA'] / st['syB']:.3f} predicted r={st['sgA'] / st['sgB']:.3f} ({seconds[-1]:.1f}s)",
              flush=True)
        del image, seg, probs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if len(rows) % 5 == 0:
            S.save_stats(out, S.stack_cases(rows))      # partial save: a lost session keeps its progress

    stats = S.stack_cases(rows)
    S.check_stats(stats)
    S.save_stats(out, stats)
    log = {"n_patients": S.n_cases(stats), "fold": args.fold, "checkpoint": args.checkpoint,
           "model_dir": args.model_dir, "tta_mirroring": args.tta, "device": str(device),
           "seconds_per_patient_mean": float(np.mean(seconds)) if seconds else None,
           "nnunetv2": nnunet_version, "torch": torch.__version__,
           "region_A_classes": list(extract.NECROSIS_CLASSES), "region_B_classes": list(extract.TUMOUR_CLASSES)}
    out.with_suffix(".json").write_text(json.dumps(log, indent=2))
    print(f"saved {S.n_cases(stats)} patients to {out}")


if __name__ == "__main__":
    main()
