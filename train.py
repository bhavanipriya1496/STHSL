import os
import torch
import numpy as np
import time

from engine import trainer
from Params import args
from utils import seed_torch, makePrint


def run_one_seed(seed: int):
    """
    Runs a full training loop for a single seed and saves checkpoints into:
      {args.save}/{args.data}/seed_{seed}/
    """
    seed_torch(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = trainer(device)

    print(f"\n===== SEED {seed} =====", flush=True)
    print("start training...", flush=True)

    train_time = []
    bestRes = None
    eval_bestRes = dict()

    if args.eval_metrics in ("error", "all"):
        eval_bestRes["RMSE"] = 1e6
        eval_bestRes["MAE"]  = 1e6
        eval_bestRes["MAPE"] = 1e6

    if args.eval_metrics in ("accuracy", "all"):
        eval_bestRes["MicroF1"] = -1.0
        eval_bestRes["MacroF1"] = -1.0
        eval_bestRes["ACC_SCORE"] = -1.0

    update = False

    # Save checkpoints under a per-seed folder (prevents overwriting across seeds)
    seed_dir = os.path.join(args.save, args.data, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    for i in range(1, args.epoch + 1):
        t1 = time.time()

        metrics, metrics1 = engine.train()

        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (
            i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1
        )
        print(ret)

        test = (i % args.tstEpoch == 0)
        if test:
            res_eval = engine.eval(True, True)

            if args.eval_metrics in ("accuracy", "all"):
                # maximize MicroF1 + MacroF1
                micro = res_eval["MicroF1"]
                macro = res_eval["MacroF1"]
                val_score = micro + macro
                best_score = eval_bestRes["ACC_SCORE"]
                if val_score > best_score:
                    print("Val (MicroF1+MacroF1) increase from %.4f to %.4f" % (best_score, val_score))
                    eval_bestRes["ACC_SCORE"] = val_score
                    eval_bestRes["MicroF1"] = micro
                    eval_bestRes["MacroF1"] = macro
                    update = True
            elif args.eval_metrics == "error":
                # minimize RMSE+MAE
                val_metrics = res_eval["RMSE"] + res_eval["MAE"]
                val_best_metrics = eval_bestRes["RMSE"] + eval_bestRes["MAE"]
                if val_metrics < val_best_metrics:
                    print("Val metrics decrease from %.4f to %.4f" % (val_best_metrics, val_metrics))
                    eval_bestRes["RMSE"] = res_eval["RMSE"]
                    eval_bestRes["MAE"]  = res_eval["MAE"]
                    update = True
            else:
                # none: do not do early stopping updates based on eval metrics
                pass

            reses = engine.eval(False, True)

            # Build checkpoint tag depending on available metrics
            # add seed into checkpoint name
            tag_parts = [f"_seed_{seed}", f"_epoch_{i}"]

            if args.eval_metrics in ("error", "all"):
                tag_parts.append(f"_MAE_{round(reses['MAE'], 2)}")
                tag_parts.append(f"_MAPE_{round(reses['MAPE'], 2)}")

            if args.eval_metrics in ("accuracy", "all"):
                tag_parts.append(f"_MicroF1_{round(reses['MicroF1'], 4)}")

            ckpt_name = "".join(tag_parts) + ".pth"

            torch.save(engine.model.state_dict(), os.path.join(seed_dir, ckpt_name))

            if update:
                print(makePrint('Test', i, reses))
                if args.eval_metrics in ("accuracy", "all"):
                    print(f"MacroF1={reses['MacroF1']:.4f} MicroF1={reses['MicroF1']:.4f} AP={reses['AP']:.4f}")
                bestRes = reses
                update = False

        print()
        t2 = time.time()
        train_time.append(t2 - t1)

    print(makePrint('Best', args.epoch, bestRes))
    return bestRes


def main():
    # Option 1: if there is added args.seeds in Params.py, this will use it.
    # Option 2: otherwise fallback to a default list.
    seeds = getattr(args, "seeds", None)
    if seeds is None:
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    all_best = {}
    for s in seeds:
        all_best[int(s)] = run_one_seed(int(s))

    print("\n===== SUMMARY: best result per seed =====")
    for s in sorted(all_best.keys()):
        print(f"seed={s}: {all_best[s]}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))