"""
Evaluation script for Flood U-Net model on unseen holdout geographic scenes.
Computes IoU, Precision, Recall, and F1 per class and per geographic region.
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from model.unet import load_flood_model
from model.dataset import FloodDataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("model.eval")

def evaluate_holdout(weights_path: str = "model/weights/kuro_siwo_unet.pt", holdout_region: str = "Nepal-Himalayas"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_flood_model(weights_path, device=str(device))

    dataset = FloodDataset(split="test", region_holdout="nepal")
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    class_names = ["Background", "Permanent Water", "Flood Water"]
    confusion_matrix = np.zeros((3, 3), dtype=np.int64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y.numpy()

            for t, p in zip(y_true.flatten(), preds.flatten()):
                if 0 <= t < 3 and 0 <= p < 3:
                    confusion_matrix[t, p] += 1

    logger.info(f"\n================ HOLD-OUT EVALUATION REPORT: {holdout_region.upper()} ================")
    results = {}
    for i, name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        iou = tp / (tp + fp + fn + 1e-7)
        prec = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-7)

        results[name] = {"IoU": round(float(iou), 4), "Precision": round(float(prec), 4), "Recall": round(float(rec), 4), "F1": round(float(f1), 4)}
        logger.info(f"Class [{name:16s}]: IoU: {iou:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

    mean_iou = np.mean([v["IoU"] for v in results.values()])
    flood_iou = results["Flood Water"]["IoU"]
    logger.info(f"--------------------------------------------------------------------------------")
    logger.info(f"Mean IoU across all classes : {mean_iou:.3f}")
    logger.info(f"Flood Water IoU on Holdout  : {flood_iou:.3f}")
    logger.info(f"================================================================================\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="model/weights/kuro_siwo_unet.pt")
    parser.add_argument("--region", type=str, default="Nepal-Himalayas")
    args = parser.parse_args()

    evaluate_holdout(weights_path=args.weights, holdout_region=args.region)
