"""Kaggle side: train nnU-Net on one fold WITHOUT nnU-Net's own final validation.

`nnUNetv2_train` always finishes with a validation pass over the held-out patients (with mirroring,
and it writes segmentations to disk). We do not need it: scripts/extract_stats.py predicts the same
held-out patients itself. This script runs exactly what `nnUNetv2_train` runs for one GPU
(get_trainer_from_args, maybe_load_checkpoint, trainer.run_training) and stops there. The final
checkpoint (checkpoint_final.pth) is written by the trainer at the end of run_training().

Usage:   python scripts/train_no_validation.py --trainer nnUNetTrainer_100epochs --fold 0
Resume:  add --continue-training   (nnU-Net saves checkpoint_latest.pth every 50 epochs)

The environment variables nnUNet_raw, nnUNet_preprocessed and nnUNet_results must be set before
launching (the notebook does it), because nnU-Net reads them when it is imported.
"""
import argparse

import torch
import torch.backends.cudnn as cudnn
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="1", help="dataset id (1 = Dataset001_BrainTumour)")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--trainer", default="nnUNetTrainer", help="e.g. nnUNetTrainer_5epochs, nnUNetTrainer_100epochs")
    p.add_argument("--plans", default="nnUNetPlans")
    p.add_argument("--continue-training", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_trainer_from_args(str(args.dataset), args.configuration, args.fold, args.trainer,
                                    args.plans, args.continue_training, device=device)
    maybe_load_checkpoint(trainer, args.continue_training, False, None)
    if device.type == "cuda":
        cudnn.deterministic = False
        cudnn.benchmark = True
    trainer.run_training()
    print("TRAINING FINISHED. Final checkpoint:", trainer.output_folder)


if __name__ == "__main__":
    main()
