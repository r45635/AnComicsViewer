#!/usr/bin/env python3
"""
Surveillance de l'entraînement Multi-BD Enhanced v2
Suit les métriques et progrès en temps réel
"""

import time
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys

def monitor_training():
    """Surveille les métriques d'entraînement."""
    print("📊 Surveillance Entraînement Multi-BD Enhanced v2")
    print("=" * 60)
    
    results_dir = Path("runs/detect/multibd_enhanced_v2")
    results_file = results_dir / "results.csv"
    
    if not results_dir.exists():
        print("⏳ En attente du démarrage de l'entraînement...")
        
    last_epoch = -1
    
    while True:
        try:
            if results_file.exists():
                # Lire les résultats
                df = pd.read_csv(results_file)
                df.columns = df.columns.str.strip()  # Nettoyer les noms de colonnes
                
                current_epoch = len(df) - 1
                
                if current_epoch > last_epoch:
                    last_epoch = current_epoch
                    
                    # Dernière époque
                    latest = df.iloc[-1]
                    
                    print(f"\n🎯 Époque {current_epoch + 1}/150")
                    print(f"   📈 Train Loss: {latest.get('train/box_loss', 0):.4f}")
                    print(f"   📉 Val Loss: {latest.get('val/box_loss', 0):.4f}")
                    print(f"   🎪 mAP50: {latest.get('metrics/mAP50(B)', 0):.4f}")
                    print(f"   🎯 mAP50-95: {latest.get('metrics/mAP50-95(B)', 0):.4f}")
                    print(f"   ⚡ Precision: {latest.get('metrics/precision(B)', 0):.4f}")
                    print(f"   🔍 Recall: {latest.get('metrics/recall(B)', 0):.4f}")
                    
                    # Meilleure performance jusqu'à présent
                    best_map50 = df['metrics/mAP50(B)'].max()
                    best_epoch = df['metrics/mAP50(B)'].idxmax()
                    
                    print(f"   🏆 Meilleur mAP50: {best_map50:.4f} (époque {best_epoch + 1})")
                    
                    # Graphique de progression simple
                    if current_epoch >= 5 and current_epoch % 10 == 0:
                        create_progress_plot(df, current_epoch)
                        
            else:
                print("⏳ Fichier de résultats non encore créé...")
                
        except Exception as e:
            print(f"⚠️  Erreur lecture métriques: {e}")
            
        time.sleep(30)  # Vérifier toutes les 30 secondes

def create_progress_plot(df, current_epoch):
    """Crée un graphique de progression."""
    try:
        plt.figure(figsize=(12, 8))
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(df.index, df['train/box_loss'], label='Train Loss')
        plt.plot(df.index, df['val/box_loss'], label='Val Loss')
        plt.title('Loss Evolution')
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # mAP
        plt.subplot(2, 2, 2)
        plt.plot(df.index, df['metrics/mAP50(B)'], label='mAP50')
        plt.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95')
        plt.title('mAP Evolution')
        plt.xlabel('Époque')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        
        # Precision/Recall
        plt.subplot(2, 2, 3)
        plt.plot(df.index, df['metrics/precision(B)'], label='Precision')
        plt.plot(df.index, df['metrics/recall(B)'], label='Recall')
        plt.title('Precision/Recall Evolution')
        plt.xlabel('Époque')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        # F1 Score (calculé)
        plt.subplot(2, 2, 4)
        precision = df['metrics/precision(B)']
        recall = df['metrics/recall(B)']
        f1 = 2 * (precision * recall) / (precision + recall)
        plt.plot(df.index, f1, label='F1 Score', color='green')
        plt.title('F1 Score Evolution')
        plt.xlabel('Époque')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Sauver le graphique
        plot_path = f"runs/detect/multibd_enhanced_v2/progress_epoch_{current_epoch}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Graphique sauvé: {plot_path}")
        
    except Exception as e:
        print(f"⚠️  Erreur création graphique: {e}")

def print_final_summary():
    """Affiche un résumé final quand l'entraînement est terminé."""
    results_dir = Path("runs/detect/multibd_enhanced_v2")
    results_file = results_dir / "results.csv"
    
    if not results_file.exists():
        print("❌ Pas de résultats trouvés")
        return
        
    try:
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()
        
        print("\n🎉 Résumé Final Multi-BD Enhanced v2")
        print("=" * 50)
        
        # Meilleures métriques
        best_map50 = df['metrics/mAP50(B)'].max()
        best_map50_epoch = df['metrics/mAP50(B)'].idxmax()
        best_map50_95 = df['metrics/mAP50-95(B)'].max()
        
        final_epoch = len(df) - 1
        final_map50 = df.iloc[-1]['metrics/mAP50(B)']
        final_map50_95 = df.iloc[-1]['metrics/mAP50-95(B)']
        
        print(f"📊 Performances Finales:")
        print(f"   • Époques complétées: {final_epoch + 1}")
        print(f"   • mAP50 final: {final_map50:.4f}")
        print(f"   • mAP50-95 final: {final_map50_95:.4f}")
        
        print(f"\n🏆 Meilleures Performances:")
        print(f"   • Meilleur mAP50: {best_map50:.4f} (époque {best_map50_epoch + 1})")
        print(f"   • Meilleur mAP50-95: {best_map50_95:.4f}")
        
        # Comparaison avec la version précédente
        print(f"\n📈 Amélioration vs v1:")
        print(f"   • v1 mAP50: 91.1%")
        print(f"   • v2 mAP50: {best_map50*100:.1f}%")
        
        if best_map50 > 0.911:
            improvement = (best_map50 - 0.911) * 100
            print(f"   • Amélioration: +{improvement:.1f}% 🎉")
        else:
            decline = (0.911 - best_map50) * 100
            print(f"   • Régression: -{decline:.1f}% ⚠️")
            
        # Fichiers de modèle
        best_model = results_dir / "weights" / "best.pt"
        last_model = results_dir / "weights" / "last.pt"
        
        print(f"\n📁 Modèles Sauvés:")
        if best_model.exists():
            print(f"   ✅ Meilleur modèle: {best_model}")
        if last_model.exists():
            print(f"   ✅ Dernier modèle: {last_model}")
            
        print(f"\n🔧 Prochaines Étapes:")
        print(f"   1. Copier le meilleur modèle:")
        print(f"      cp {best_model} detectors/models/multibd_enhanced_v2.pt")
        print(f"   2. Mettre à jour le détecteur dans AnComicsViewer")
        print(f"   3. Tester les nouvelles performances")
        
    except Exception as e:
        print(f"❌ Erreur lecture résumé: {e}")

def main():
    """Point d'entrée principal."""
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_final_summary()
        return
        
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n⏹️  Surveillance interrompue")
        print_final_summary()

if __name__ == "__main__":
    main()
