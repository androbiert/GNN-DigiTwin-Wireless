import json
import re

def decrement_gnb_name(name):
    """
    Prend une chaîne comme 'gnb1', extrait le chiffre, fait -1 
    et retourne 'gnb0'. Gère aussi 'none' ou les chaînes vides.
    """
    if not name or name.lower() == "none":
        return name
    
    # Recherche du texte suivi de chiffres (ex: gnb et 1)
    match = re.match(r"([a-zA-Z]+)(\d+)", name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return f"{prefix}{max(0, number - 1)}" # max(0,...) évite les nombres négatifs
    
    return name

def process_simulation_data(input_filename, output_filename, tx_power, sched_discipline, queue_size):
    try:
        # 1. Charger les données JSON
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Parcourir chaque timestamp
        for entry in data:
            if "nodes" in entry:
                for node in entry["nodes"]:
                    node_id = node.get("id", "")

                    # --- Traitement commun : Décrémenter serving_gnb ---
                    if "serving_gnb" in node:
                        node["serving_gnb"] = decrement_gnb_name(node["serving_gnb"])

                    # --- Traitement spécifique : gNB ---
                    if node_id.startswith("gnb"):
                        # Ajout des nouveaux paramètres
                        node["tx_power"] = tx_power
                        node["scheduling_discipline"] = sched_discipline
                        node["queue_size"] = queue_size
                        
                        # Suppression des attributs SINR pour les gNB
                        node.pop("sinr_dl", None)
                        node.pop("sinr_ul", None)
                    
                    # --- Traitement spécifique : UE ---
                    elif node_id.startswith("ue"):
                        node["qsize"] = queue_size

        # 3. Sauvegarder le résultat
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Fichier traité avec succès ! Résultat : {output_filename}")

    except Exception as e:
        print(f"Erreur : {e}")

# --- PARAMÈTRES DE CONFIGURATION ---
TX_POWER_VAL = 0.01
SCHED_DISCIPLINE_VAL = "PF"
QUEUE_SIZE_VAL = "100KiB"

# --- EXÉCUTION ---
process_simulation_data(
    input_filename='network_state.json',        
    output_filename='data.json', 
    tx_power=TX_POWER_VAL, 
    sched_discipline=SCHED_DISCIPLINE_VAL, 
    queue_size=QUEUE_SIZE_VAL
)
