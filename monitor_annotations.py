#!/usr/bin/env python3
"""
Monitor Annotation Progress
Surveille le progrès d'annotation en temps réel.
"""

import time
import json
from pathlib import Path
import os

def clear_screen():
    """Efface l'écran."""
    os.system('clear' if os.name == 'posix' else 'cls')

def count_annotations():
    """Compte les annotations par catégorie."""
    
    labels_dir = Path("dataset/labels/train")
    
    golden_city_total = 44
    tintin_total = 66
    
    golden_city_annotated = 0
    tintin_annotated = 0
    total_panels = 0
    total_text = 0
    
    # Compter les fichiers et analyser le contenu
    for json_file in labels_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Compter par source
            if json_file.name.startswith("tintin_"):
                tintin_annotated += 1
            else:
                golden_city_annotated += 1
            
            # Compter les objets annotés
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower()
                if label == 'panel':
                    total_panels += 1
                elif label == 'text':
                    total_text += 1
                    
        except Exception:
            continue
    
    return {
        'golden_city': {'annotated': golden_city_annotated, 'total': golden_city_total},
        'tintin': {'annotated': tintin_annotated, 'total': tintin_total},
        'objects': {'panels': total_panels, 'text': total_text}
    }

def display_progress():
    """Affiche le progrès en temps réel."""
    
    while True:
        clear_screen()
        
        stats = count_annotations()
        
        print("🎯 ANNOTATION PROGRESS - MONITORING EN TEMPS RÉEL")
        print("=" * 55)
        print(f"⏰ Dernière mise à jour: {time.strftime('%H:%M:%S')}")
        print()
        
        # Progrès par source
        gc = stats['golden_city']
        tt = stats['tintin']
        
        gc_pct = gc['annotated'] / gc['total'] * 100
        tt_pct = tt['annotated'] / tt['total'] * 100
        total_pct = (gc['annotated'] + tt['annotated']) / (gc['total'] + tt['total']) * 100
        
        print("📊 PROGRÈS PAR SOURCE:")
        print("=" * 25)
        print(f"🏛️  Golden City: {gc['annotated']:2d}/{gc['total']} ({gc_pct:5.1f}%) {'▓' * int(gc_pct//5)}")
        print(f"🎨 Tintin:      {tt['annotated']:2d}/{tt['total']} ({tt_pct:5.1f}%) {'▓' * int(tt_pct//5)}")
        print(f"📈 Total:       {gc['annotated']+tt['annotated']:2d}/{gc['total']+tt['total']} ({total_pct:5.1f}%) {'▓' * int(total_pct//5)}")
        print()
        
        # Objets annotés
        objects = stats['objects']
        print("🎯 OBJETS ANNOTÉS:")
        print("=" * 20)
        print(f"📦 Panels: {objects['panels']}")
        print(f"💬 Text:   {objects['text']}")
        print(f"📊 Total:  {objects['panels'] + objects['text']}")
        print()
        
        # Estimations
        if tt['annotated'] > 0:
            avg_panels_per_page = objects['panels'] / (gc['annotated'] + tt['annotated'])
            estimated_total_panels = avg_panels_per_page * (gc['total'] + tt['total'])
            print(f"📈 Estimation panels totaux: {estimated_total_panels:.0f}")
            print(f"⚡ Moyenne panels/page: {avg_panels_per_page:.1f}")
            print()
        
        # Objectifs
        print("🎯 OBJECTIFS:")
        print("=" * 15)
        needed_for_30pct = int(0.3 * (gc['total'] + tt['total'])) - (gc['annotated'] + tt['annotated'])
        needed_for_50pct = int(0.5 * (gc['total'] + tt['total'])) - (gc['annotated'] + tt['annotated'])
        
        if needed_for_30pct > 0:
            print(f"🚀 Pour 30%: {needed_for_30pct} annotations de plus")
        else:
            print("✅ 30% atteint! Prêt pour test d'entraînement")
            
        if needed_for_50pct > 0:
            print(f"🏆 Pour 50%: {needed_for_50pct} annotations de plus")
        else:
            print("🎉 50% atteint! Excellent pour entraînement")
        
        print()
        print("💡 Ctrl+C pour arrêter le monitoring")
        print("🔄 Mise à jour automatique toutes les 5 secondes")
        
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n👋 Monitoring arrêté")
            break

if __name__ == "__main__":
    try:
        display_progress()
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"❌ Erreur: {e}")
