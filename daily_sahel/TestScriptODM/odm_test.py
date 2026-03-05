print("entering odm_test.py")
import time
from pyodm import Node
import os
import requests

# Connexion à NodeODM
node = Node("localhost", 3000)
# Infos générales sur le noeud NodeODM


# Répertoire contenant tes images (chemin absolu recommandé)
image_dir = "/media/rfernandez/Crucial X9/Test_anatomics_tmp/Test_reconstruct_ODM/2022/17h/MS_STR_short"  # ← à adapter
image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(".tif")]

# Paramètres du traitement
options = {
    "dsm": True,                 # Générer un MNT
    "orthophoto-resolution": 1,  # Résolution en cm/pixel
    "orthophoto-bands": "all",            # inclure toutes les bandes dans l'orthophoto multi-bande
    "use-3dmesh": True          # Désactiver le maillage 3D (optionnel)
}




def print_summary(task):
    print("\n[Résumé de la tâche ODM]")

    try:
        info = task.info()
        status = info.status.name
        duration = info.processing_time
        progress = info.progress
        failed = info.status.is_failed()

        print(f"  Statut final      : {status}")
        print(f"  Progression finale: {progress:.1f}%")
        if duration:
            print(f"  Temps total       : {duration / 60:.1f} minutes")
        else:
            print("  Temps total       : inconnu")

        if failed:
            print("  ⚠️  Le traitement a échoué.")
            print("  ▶️  Dernier message :")
            print(f"     {info.status.message}")
        else:
            assets = task.list_assets()
            print(f"  Fichiers produits : {len(assets)}")
            for name in sorted(assets.keys()):
                print(f"    • {name}")

    except Exception as e:
        print(f"  ⚠️  Erreur lors du résumé : {e}")



# Lancer la tâche
task = node.create_task(image_list, options)

# Attendre la fin du traitement
print("Traitement en cours...")

start_time = time.time()
last_progress = -1
line_num = 0
last_status_time = time.time()
STATUS_INTERVAL = 30  # Afficher le statut toutes les 30 secondes

# Patterns à filtrer (ne pas afficher)
SKIP_PATTERNS = [
    "INFO: -------",  # Lignes de séparation
    "Can't use features matching, will use ECC",
    "Pyramid levels:",
    "Could not compute ECC warp_matrix at pyramid level"
]

def should_skip_line(line):
    """Retourne True si la ligne doit être ignorée"""
    # Ignorer les lignes vides
    if not line.strip():
        return True
    
    # Ignorer les lignes contenant un pattern à filtrer
    for pattern in SKIP_PATTERNS:
        if pattern in line:
            return True
    
    # Ignorer les lignes commençant par une date (format: YYYY-MM-DD HH:MM:SS)
    import re
    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line):
        return True
    
    return False

while True:
    info = task.info()
    status_name = info.status.name
    progress = info.progress
    current_time = time.time()
    elapsed = current_time - start_time
    
    # Récupérer les logs via API REST directement avec timeout
    try:
        url = f"http://localhost:3000/task/{task.uuid}/output?line={line_num}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            new_lines = response.text.strip().split('\n')
            
            # Afficher les nouveaux logs après filtrage
            for line in new_lines:
                if not should_skip_line(line):
                    print(line)
            
            line_num += len(new_lines)
    except requests.Timeout:
        print("⏱ Timeout")
    except Exception as e:
        pass
    
    # Toutes les 30 secondes : pause et affichage du statut
    if current_time - last_status_time >= STATUS_INTERVAL:
        print("\n\n\n")  # 3 lignes de saut
        
        if progress > 0:
            estimated_total = elapsed / (progress / 100)
            remaining = estimated_total - elapsed
            print(f"[{status_name}] {progress:.1f}% - Écoulé: {elapsed/60:.1f}min - Restant: {remaining/60:.1f}min")
        else:
            print(f"[{status_name}] {progress:.1f}% - Écoulé: {elapsed/60:.1f}min")
        
        print("\n")
        time.sleep(2)  # Pause de 2 secondes
        last_status_time = current_time
     
    if status_name in ["COMPLETED", "FAILED", "CANCELED"]:
        break
    time.sleep(1)  # Vérifier toutes les secondes pour un affichage fluide
    print("\n")

print(f"\nTerminé en {(time.time() - start_time)/60:.1f} minutes.")
print_summary(task)

# Télécharger les résultats
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
task.download_assets(output_dir)
print(f"Résultats enregistrés dans : {output_dir}")
