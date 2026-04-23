import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
json_file_path = 'network_state.json'
output_root = 'simulation_plots'

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_plots():
    # 1. Chargement des données
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    node_records = []
    flow_records = []

    # 2. Extraction et transformation des données
    for snapshot in data:
        ts = snapshot['timestamp']
        
        # Nodes data (Position, Speed, SINR)
        for node in snapshot['nodes']:
            if "ue" in node['id']:  # On ignore le gnb pour les plots
                node_records.append({
                    'timestamp': ts,
                    'ue_id': node['id'],
                    'x': node['x'],
                    'y': node['y'],
                    'speed': node['speed'],
                    'sinr_dl': node['sinr_dl'],
                    'sinr_ul': node['sinr_ul']
                })
        
        # Flows data (Throughput, Delay, Packet Size...)
        for flow in snapshot['flows']:
            # Nettoyage de l'ID (ue[0] -> ue0) pour matcher les nodes
            ue_id = flow['dst'].replace('[', '').replace(']', '')
            flow_records.append({
                'timestamp': ts,
                'ue_id': ue_id,
                'type': flow['type'],
                'packet_size': flow['packet_size'],
                'interval': flow['interval'],
                'throughput': flow['throughput'],
                'delay': flow['delay'],
                'bler': flow['bler'],
                'packet_loss': flow['packet_loss']
            })

    df_nodes = pd.DataFrame(node_records)
    df_flows = pd.DataFrame(flow_records)

    # 3. Définition des paramètres à tracer
    node_params = ['x', 'y', 'speed', 'sinr_dl', 'sinr_ul']
    flow_params = ['packet_size', 'interval', 'throughput', 'delay', 'bler', 'packet_loss']

    # 4. Boucle de génération des plots
    print(f"Début de la génération des graphiques dans le dossier : {output_root}")

    # Plotting NODE parameters
    for param in node_params:
        param_dir = os.path.join(output_root, param)
        create_dir(param_dir)
        
        for ue in df_nodes['ue_id'].unique():
            ue_data = df_nodes[df_nodes['ue_id'] == ue]
            
            plt.figure(figsize=(10, 5))
            plt.plot(ue_data['timestamp'], ue_data[param], marker='o', linestyle='-', color='b')
            plt.title(f"Évolution de {param} pour {ue}")
            plt.xlabel("Temps (s)")
            plt.ylabel(param)
            plt.grid(True)
            
            plt.savefig(os.path.join(param_dir, f"{ue}_{param}.png"))
            plt.close()

    # Plotting FLOW parameters
    for param in flow_params:
        param_dir = os.path.join(output_root, param)
        create_dir(param_dir)
        
        for ue in df_flows['ue_id'].unique():
            ue_data = df_flows[df_flows['ue_id'] == ue]
            
            if ue_data.empty: continue

            plt.figure(figsize=(10, 5))
            plt.plot(ue_data['timestamp'], ue_data[param], marker='s', linestyle='--', color='r')
            plt.title(f"Évolution de {param} (Flow) pour {ue}")
            plt.xlabel("Temps (s)")
            plt.ylabel(param)
            plt.grid(True)
            
            plt.savefig(os.path.join(param_dir, f"{ue}_{param}.png"))
            plt.close()

    print("✅ Terminé ! Tous les dossiers ont été créés et les graphiques enregistrés.")

if __name__ == "__main__":
    generate_plots()
