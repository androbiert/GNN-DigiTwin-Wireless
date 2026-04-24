import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from wireless_gnn.model import WirelessNetFermi
from wireless_gnn.dataset import build_datasets
from wireless_gnn.train import predict

def load_model(target, checkpoint_path, device):
    """Charge le modèle avec les paramètres par défaut pour le target donné."""
    if not os.path.exists(checkpoint_path):
        print(f"[!] Attention : Le checkpoint '{checkpoint_path}' n'existe pas.")
        return None
        
    model = WirelessNetFermi(
        hidden_dim=64,
        num_heads=4,
        iterations=8,
        target=target
    ).to(device)
    
    # Charger les poids
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    print("=" * 60)
    print("  Évaluation des Modèles : Délai et Débit (10 Échantillons)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Chemins par défaut des meilleurs modèles
    delay_ckpt = "wireless_gnn/checkpoints/delay/best.pt"
    tput_ckpt  = "wireless_gnn/checkpoints/throughput/best.pt"
    
    # 1. Charger les deux modèles
    model_delay = load_model("delay", delay_ckpt, device)
    model_tput  = load_model("throughput", tput_ckpt, device)
    
    if model_delay is None and model_tput is None:
        print("Aucun modèle n'a été trouvé. Veuillez lancer l'entraînement d'abord.")
        return

    # 2. Charger le dataset (on récupère le test_ds)
    print("\nChargement des données de test...")
    # build_datasets renvoie: train_ds, val_ds, test_ds, normalizer
    _, _, test_ds, norm = build_datasets(".")
    
    if len(test_ds) == 0:
        print("Erreur : Le dataset de test est vide.")
        return

    # 3. Tirer 10 échantillons aléatoires du dataset de test
    n_samples = min(10, len(test_ds))
    # On sélectionne les indices aléatoires
    indices = random.sample(range(len(test_ds)), n_samples)
    
    print(f"\nComparaison sur {n_samples} échantillons tirés au hasard :\n")
    
    delay_true_list = []
    delay_pred_list = []
    tput_true_list = []
    tput_pred_list = []
    labels = []
    
    for i, idx in enumerate(indices, 1):
        graph = test_ds[idx]
        
        print(f"--- Échantillon {i} (Index dans le dataset: {idx}) ---")
        
        # --- DELAY ---
        if model_delay is not None:
            # Prédiction
            res_delay = predict(model_delay, graph, norm, device)
            
            # Les valeurs sont retournées en tant que tableaux (arrays) par la fonction predict
            true_d_s = res_delay["delay_true"].item() if res_delay["delay_true"].size == 1 else res_delay["delay_true"][0]
            pred_d_s = res_delay["delay_pred"].item() if res_delay["delay_pred"].size == 1 else res_delay["delay_pred"][0]
            
            # Conversion en millisecondes (ms)
            true_d_ms = true_d_s * 1000
            pred_d_ms = pred_d_s * 1000
            error_d_pct = abs(true_d_ms - pred_d_ms) / (abs(true_d_ms) + 1e-6) * 100
            
            delay_true_list.append(true_d_ms)
            delay_pred_list.append(pred_d_ms)
            
            print(f"  Délai :")
            print(f"    Vrai    = {true_d_ms:>8.3f} ms")
            print(f"    Prédit  = {pred_d_ms:>8.3f} ms")
            print(f"    Erreur  = {error_d_pct:>8.2f} %")
            
        # --- THROUGHPUT (Débit) ---
        if model_tput is not None:
            # Prédiction
            res_tput = predict(model_tput, graph, norm, device)
            
            true_t_bps = res_tput["throughput_true"].item() if res_tput["throughput_true"].size == 1 else res_tput["throughput_true"][0]
            pred_t_bps = res_tput["throughput_pred"].item() if res_tput["throughput_pred"].size == 1 else res_tput["throughput_pred"][0]
            
            # Conversion en kbps
            true_t_kbps = true_t_bps / 1000
            pred_t_kbps = pred_t_bps / 1000
            error_t_pct = abs(true_t_kbps - pred_t_kbps) / (abs(true_t_kbps) + 1e-6) * 100
            
            tput_true_list.append(true_t_kbps)
            tput_pred_list.append(pred_t_kbps)
            
            print(f"  Débit (Throughput) :")
            print(f"    Vrai    = {true_t_kbps:>8.2f} kbps")
            print(f"    Prédit  = {pred_t_kbps:>8.2f} kbps")
            print(f"    Erreur  = {error_t_pct:>8.2f} %")
        
        labels.append(f"E{i}")
        print("")

    # --- GENERATE PLOT ---
    if labels:
        x = np.arange(len(labels))
        width = 0.35

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Subplot 1: Delay
        if delay_true_list:
            ax1 = axes[0]
            ax1.bar(x - width/2, delay_true_list, width, label='Vrai (ms)', color='#4C72B0')
            ax1.bar(x + width/2, delay_pred_list, width, label='Prédit (ms)', color='#DD8452')
            ax1.set_ylabel('Délai (ms)')
            ax1.set_title('Comparaison du Délai sur 10 échantillons', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

        # Subplot 2: Throughput
        if tput_true_list:
            ax2 = axes[1]
            ax2.bar(x - width/2, tput_true_list, width, label='Vrai (kbps)', color='#4C72B0')
            ax2.bar(x + width/2, tput_pred_list, width, label='Prédit (kbps)', color='#DD8452')
            ax2.set_ylabel('Débit (kbps)')
            ax2.set_title('Comparaison du Débit sur 10 échantillons', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = "eval_samples_comparison.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Graphique de comparaison sauvegardé sous : {save_path}\n")

if __name__ == "__main__":
    main()
