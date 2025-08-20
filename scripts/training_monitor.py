#!/usr/bin/env python3
"""
Monitor d'entraînement en temps réel pour Multi-BD Enhanced v2
Affiche les métriques et les graphiques pendant l'entraînement
"""

import time
import sys
from pathlib import Path
import subprocess

def monitor_training():
    """Surveille l'entraînement en cours et affiche des métriques."""
    print("📊 Monitor d'entraînement Multi-BD Enhanced v2")
    print("=" * 50)
    
    # Chercher les dossiers d'entraînement actifs
    runs_path = Path("runs/detect")
    if not runs_path.exists():
        print("❌ Aucun entraînement en cours")
        return
    
    # Trouver le run le plus récent
    recent_runs = sorted(runs_path.glob("multibd_enhanced_v2_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not recent_runs:
        print("❌ Aucun run Multi-BD Enhanced v2 trouvé")
        return
    
    current_run = recent_runs[0]
    print(f"🎯 Monitoring: {current_run.name}")
    
    # Files à surveiller
    results_csv = current_run / "results.csv"
    
    last_size = 0
    epoch = 0
    
    print("\n⏳ En attente du démarrage de l'entraînement...")
    
    try:
        while True:
            # Vérifier si l'entraînement est toujours actif
            if results_csv.exists():
                current_size = results_csv.stat().st_size
                if current_size > last_size:
                    # Nouveau contenu dans results.csv
                    try:
                        with open(results_csv, 'r') as f:
                            lines = f.readlines()
                        
                        if len(lines) > epoch + 1:  # +1 pour l'header
                            latest_line = lines[-1].strip()
                            values = latest_line.split(',')
                            epoch = len(lines) - 1
                            
                            print(f"\n📈 Époque {epoch}")
                            if len(values) >= 4:  # Au minimum epoch, box_loss, obj_loss, cls_loss
                                try:
                                    print(f"   📦 Box Loss: {float(values[1]):.4f}")
                                    print(f"   🎯 Obj Loss: {float(values[2]):.4f}")
                                    print(f"   🏷️  Cls Loss: {float(values[3]):.4f}")
                                    
                                    # Chercher mAP si disponible (souvent en fin de ligne)
                                    if len(values) > 6:
                                        try:
                                            map50 = float(values[6])
                                            print(f"   🎯 mAP@50: {map50:.3f}")
                                        except:
                                            pass
                                except ValueError:
                                    pass
                            
                            # Estimation du temps restant
                            if epoch > 1:
                                epochs_total = 200  # Default from our config
                                remaining_epochs = epochs_total - epoch
                                print(f"   📊 Progression: {epoch}/{epochs_total} ({100*epoch/epochs_total:.1f}%)")
                    
                    except Exception as e:
                        print(f"⚠️  Erreur lecture métriques: {e}")
                    
                    last_size = current_size
            
            # Vérifier si l'entraînement est terminé
            weights_path = current_run / "weights"
            if weights_path.exists() and (weights_path / "best.pt").exists():
                print(f"\n🎉 Entraînement terminé!")
                print(f"🎯 Modèle final: {weights_path / 'best.pt'}")
                break
            
            time.sleep(10)  # Check toutes les 10 secondes
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring arrêté")

def main():
    monitor_training()

if __name__ == "__main__":
    main()
