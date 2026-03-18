"""统一实验入口（train / eval）。

用法:
  python -m experiment.run train  --config experiment/configs/fisnet_task1.json
  python -m experiment.run eval   --ckpt checkpoints/fis_net_xxx/best.pt --config ...
"""

from __future__ import annotations

import argparse
import sys

from experiment.config import load_config
from experiment.evaluate import main as eval_main
from experiment.train import run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MM-FIS 实验统一入口")
    sp = p.add_subparsers(dest="cmd", required=True)

    p_train = sp.add_parser("train", help="训练模型")
    p_train.add_argument(
        "--config", type=str,
        default="/CIL_PROJECTS/CODES/MM_FIS/experiment/configs/default.json",
        help="JSON 配置文件路径",
    )
    p_train.add_argument(
        "--device", type=str, default=None,
        help="覆盖配置中的 device，如 cuda、cuda:0、cuda:1；不传则用 config 内 train.device",
    )

    p_eval = sp.add_parser("eval", help="评估已有 checkpoint")
    p_eval.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    p_eval.add_argument(
        "--config", type=str,
        default="/CIL_PROJECTS/CODES/MM_FIS/experiment/configs/default.json",
    )
    p_eval.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p_eval.add_argument(
        "--device", type=str, default=None,
        help="覆盖配置中的 device，如 cuda:0；不传则用 config 内 train.device",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "train":
        cfg = load_config(args.config)
        if args.device is not None:
            cfg.train.device = args.device
        run_training(cfg)
        return

    eval_argv = ["evaluate.py", "--ckpt", args.ckpt, "--config", args.config, "--split", args.split]
    if args.device is not None:
        eval_argv += ["--device", args.device]
    sys.argv = eval_argv
    eval_main()


if __name__ == "__main__":
    main()
