import json
import re

def decrement_gnb_name(name):
    """
    Transforme 'gnb1' en 'gnb0', 'gnb2' en 'gnb1', etc.
    """
    if not name or name.lower() == "none":
        return name
    
    match = re.match(r"([a-zA-Z]+)(\d+)", name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return f"{prefix}{max(0, number - 1)}"
    
    return name

def process_simulation_data(input_filename, output_filename, tx_power, sched_discipline, queue_size):
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            # --- 1. TRAITEMENT DES NODES ---
            if "nodes" in entry:
                for node in entry["nodes"]:
                    node_id = node.get("id", "")

                    # Mise à jour universelle du serving_gnb
                    if "serving_gnb" in node:
                        node["serving_gnb"] = decrement_gnb_name(node["serving_gnb"])

                    # Cas gNB
                    if node_id.startswith("gnb"):
                        node["tx_power"] = tx_power
                        node["scheduling_discipline"] = sched_discipline
                        node["queue_size"] = queue_size
                        # Supprimer SINR pour gNB
                        node.pop("sinr_dl", None)
                        node.pop("sinr_ul", None)
                    
                    # Cas UE
                    elif node_id.startswith("ue"):
                        node["qsize"] = queue_size
                        # Supprimer app et bler si présents dans le node
                        node.pop("app", None)
                        node.pop("bler", None)

            # --- 2. TRAITEMENT DES FLOWS (où se trouvent app et bler) ---
            if "flows" in entry:
                for flow in entry["flows"]:
                    # On supprime 'app' et 'bler' pour tous les flux liés aux UEs
                    flow.pop("app", None)
                    flow.pop("bler", None)

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Modification terminée avec succès ! Fichier : {output_filename}")

    except Exception as e:
        print(f"Erreur durant le traitement : {e}")

# --- PARAMÈTRES ---
TX_POWER = 0.01
SCHEDULING = "PF"
QUEUE_SIZE = "100KiB"

# --- LANCEMENT ---
process_simulation_data(
    input_filename='network_state.json', 
    output_filename='data.json', 
    tx_power=TX_POWER, 
    sched_discipline=SCHEDULING, 
    queue_size=QUEUE_SIZE
)
