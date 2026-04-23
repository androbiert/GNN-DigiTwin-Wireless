import json
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
FILE_PATH = 'network_state.json'
OUTPUT_ROOT = 'plots_network'

def clean_ue_id(ue_name):
    """Transforme 'ue[0]' en 'ue0' pour l'uniformité"""
    return ue_name.replace('[', '').replace(']', '')

def create_plots():
    # 1. Chargement des données
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    # Structure : data_store[metric][ue_id] = {'t': [...], 'v': [...]}
    data_store = {}

    # 2. Extraction des données
    for entry in data:
        ts = entry['timestamp']
        
        # --- NODES ---
        for node in entry['nodes']:
            if node['id'].startswith('ue'):
                ue_id = node['id']
                metrics = ['sinr_dl', 'sinr_ul', 'speed', 'x', 'y']
                
                for m in metrics:
                    if m in node:
                        data_store.setdefault(m, {})
                        data_store[m].setdefault(ue_id, {'t': [], 'v': []})
                        
                        data_store[m][ue_id]['t'].append(ts)
                        data_store[m][ue_id]['v'].append(node[m])

        # --- FLOWS ---
        for flow in entry['flows']:
            ue_id = clean_ue_id(flow['dst'])
            flow_metrics = ['throughput', 'delay', 'bler', 'packet_loss', 'rlcDelay', 'harqTxAttempts']
            
            for m in flow_metrics:
                if m in flow:
                    data_store.setdefault(m, {})
                    data_store[m].setdefault(ue_id, {'t': [], 'v': []})
                    
                    data_store[m][ue_id]['t'].append(ts)
                    data_store[m][ue_id]['v'].append(flow[m])

    # 3. Création des dossiers
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for metric, ues in data_store.items():
        metric_folder = os.path.join(OUTPUT_ROOT, metric)
        os.makedirs(metric_folder, exist_ok=True)
        
        print(f"Génération des plots pour : {metric}")

        for ue_id, values in ues.items():
            plt.figure(figsize=(10, 6))
            
            # ✅ Ligne uniquement (PAS de marker)
            plt.plot(values['t'], values['v'], linestyle='-', linewidth=0.8, label=ue_id)

            plt.title(f"{metric} - {ue_id}")
            plt.xlabel("Temps (s)")
            plt.ylabel(metric)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            plt.savefig(os.path.join(metric_folder, f"{ue_id}.png"))
            plt.close()

    print(f"\n✔ Terminé ! Graphiques dans : {OUTPUT_ROOT}")

if __name__ == "__main__":
    create_plots()
