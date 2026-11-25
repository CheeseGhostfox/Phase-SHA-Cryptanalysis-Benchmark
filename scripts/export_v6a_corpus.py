"""Offline verified V6A Reverse-CoT trajectory exporter for V7C Stage 2 warm-start SFT.

Strictly runs `v6a.v6a_engine.solve_single_digest_v6a` in `free_msg` mode with
`allow_peel_back=False` ONLY OFFLINE before V7 training, saving records only after
independent `src.sha256_ref.sha256_digest` verification. Never imported at eval time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sha256_ref import MASK32, sha256_digest  # noqa: E402
from v6a.v6a_engine import _sample_free_msg_w16, solve_single_digest_v6a  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export verified offline V6A trajectories for V7C")
    p.add_argument("--rounds", type=str, default="4,5", help="Comma-separated rounds (default: 4,5)")
    p.add_argument("--per_round", type=int, default=32, help="Target digests attempted per round")
    p.add_argument("--num_trajs", type=int, default=128)
    p.add_argument("--max_ticks", type=int, default=80)
    p.add_argument("--max_seconds", type=float, default=6.0)
    p.add_argument("--out", type=str, default="runs/v7c_v6a_corpus.json")
    p.add_argument("--seed", type=int, default=6400)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dev = torch.device(args.device)
    rounds_list = [int(x.strip()) for x in args.rounds.split(",") if x.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    t_start = time.monotonic()

    for r in rounds_list:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(args.seed + r * 97)
        hits = 0
        for idx in range(args.per_round):
            w_sample = _sample_free_msg_w16(1, rounds=r, device=dev, gen=gen)
            target_d = sha256_digest(w_sample, rounds=r).squeeze(0)
            res = solve_single_digest_v6a(
                digest=target_d,
                rounds=r,
                domain_mode="free_msg",
                allow_peel_back=False,
                num_trajs=args.num_trajs,
                max_ticks=args.max_ticks,
                max_seconds=args.max_seconds,
                device=dev,
                seed=args.seed + r * 10000 + idx,
            )
            hit_w = res.get("hit_w16")
            if bool(res.get("solved")) and hit_w is not None:
                w16_t = torch.tensor([int(x) & MASK32 for x in hit_w], dtype=torch.int64, device=dev).unsqueeze(0)
                if r < 16:
                    w16_t[:, r:] = 0
                d_check = sha256_digest(w16_t, rounds=r).squeeze(0)
                if bool((d_check == target_d).all().item()):
                    hits += 1
                    records.append(
                        {
                            "rounds": int(r),
                            "w16": [int(x) for x in w16_t.squeeze(0).tolist()],
                            "digest": [int(x) for x in target_d.tolist()],
                        }
                    )
        print(
            f"[export_v6a_corpus] R={r} solved={hits}/{args.per_round} "
            f"elapsed={time.monotonic() - t_start:.1f}s",
            flush=True,
        )

    payload = {
        "schema": "v7c_offline_v6a_corpus_v1",
        "rounds": rounds_list,
        "count": len(records),
        "records": records,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[export_v6a_corpus] saved {len(records)} verified records -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
