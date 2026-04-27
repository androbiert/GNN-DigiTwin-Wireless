import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from wireless_gnn.model import WirelessNetFermi
from wireless_gnn.dataset import build_datasets, collate_fn
from wireless_gnn.train import predict

def main():
    print("=" * 60)
    print("  Évaluation Générale du Modèle : Délai")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "wireless_gnn/checkpoints/delay/best.pt"

    if not os.path.exists(checkpoint_path):
        print(f"[!] Checkpoint introuvable : {checkpoint_path}")
        return

    # Load Model
    model = WirelessNetFermi(
        hidden_dim=64,
        num_heads=4,
        iterations=8,
        target="delay"
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    print("Modèle 'Delay' chargé avec succès.\n")

    # Load Data
    _, _, test_ds, norm = build_datasets(".")
    if len(test_ds) == 0:
        print("Erreur : Le dataset de test est vide.")
        return

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print(f"Évaluation sur le test set complet ({len(test_ds)} échantillons)...")

    true_list = []
    pred_list = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            for graph in batch:
                res = predict(model, graph, norm, device)
                true_vals = res["delay_true"] * 1000  # convert to ms
                pred_vals = res["delay_pred"] * 1000  # convert to ms
                
                # append each flow's true and pred
                true_list.extend(true_vals.tolist())
                pred_list.extend(pred_vals.tolist())

    true_arr = np.array(true_list)
    pred_arr = np.array(pred_list)

    # Compute metrics
    mae = np.mean(np.abs(true_arr - pred_arr))
    rmse = np.sqrt(np.mean((true_arr - pred_arr)**2))
    mape = np.mean(np.abs((true_arr - pred_arr) / (np.abs(true_arr) + 1e-6))) * 100
    
    # R2 Score
    ss_res = np.sum((true_arr - pred_arr)**2)
    ss_tot = np.sum((true_arr - np.mean(true_arr))**2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))

    print("\n" + "=" * 60)
    print("RÉSULTATS DE L'ÉVALUATION GÉNÉRALE (Test Set)")
    print("=" * 60)
    print(f"  Nombre de graphes (snapshots) : {len(test_ds)}")
    print(f"  Nombre total de flux évalués  : {len(true_arr)}")
    print(f"  MAE (Mean Abs Error)          : {mae:.3f} ms")
    print(f"  RMSE                          : {rmse:.3f} ms")
    print(f"  MAPE                          : {mape:.3f} %")
    print(f"  R² Score                      : {r2:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
