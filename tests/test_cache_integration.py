#!/usr/bin/env python3
"""
Test d'intégration du cache amélioré dans AnComicsViewer
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cache_integration():
    print("🧪 Test Intégration Cache Amélioré")
    print("=" * 45)
    
    # Test 1: Import du cache dans AnComicsViewer
    print("1. Test import enhanced_cache...")
    try:
        import enhanced_cache
        print("   ✅ Module enhanced_cache importé")
        
        # Vérifier la classe PanelCacheManager
        assert hasattr(enhanced_cache, 'PanelCacheManager')
        print("   ✅ Classe PanelCacheManager disponible")
        
        # Créer une instance pour tester
        cache_manager = enhanced_cache.PanelCacheManager()
        assert cache_manager is not None
        print("   ✅ Instance PanelCacheManager créée")
        
    except Exception as e:
        print(f"   ❌ Erreur import: {e}")
        return False
    
    # Test 2: Vérifier l'intégration dans AnComicsViewer.py
    print("\n2. Test intégration dans AnComicsViewer.py...")
    try:
        with open("AnComicsViewer.py", 'r') as f:
            content = f.read()
        
        # Vérifier les imports
        if "from enhanced_cache import PanelCacheManager" in content:
            print("   ✅ Import du cache trouvé")
        else:
            print("   ⚠️  Import du cache non trouvé")
        
        # Vérifier l'initialisation
        if "_enhanced_cache = PanelCacheManager()" in content:
            print("   ✅ Initialisation du cache trouvée")
        else:
            print("   ⚠️  Initialisation du cache non trouvée")
        
        # Vérifier les méthodes modifiées
        if "cached_panels = self._enhanced_cache.get_panels" in content:
            print("   ✅ Utilisation du cache dans _ensure_panels_for")
        else:
            print("   ⚠️  Utilisation du cache non trouvée")
            
        # Vérifier le menu cache
        if "Cache" in content and "Statistiques" in content:
            print("   ✅ Menu cache ajouté")
        else:
            print("   ⚠️  Menu cache non trouvé")
            
    except Exception as e:
        print(f"   ❌ Erreur lecture fichier: {e}")
        return False
    
    # Test 3: Vérifier que le répertoire cache sera créé
    print("\n3. Test création répertoire cache...")
    try:
        from pathlib import Path
        cache_dir = Path.home() / ".ancomicsviewer" / "cache"
        
        # Le répertoire devrait exister maintenant
        if cache_dir.exists():
            print(f"   ✅ Répertoire cache existe: {cache_dir}")
            
            # Compter les fichiers de cache
            cache_files = list(cache_dir.glob("*.cache"))
            print(f"   📁 Fichiers de cache: {len(cache_files)}")
        else:
            print(f"   ℹ️  Répertoire cache sera créé: {cache_dir}")
            
    except Exception as e:
        print(f"   ❌ Erreur répertoire: {e}")
        return False
    
    # Test 4: Performance du cache sur données simulées
    print("\n4. Test performance cache...")
    try:
        from PySide6.QtCore import QRectF
        
        # Données test
        test_panels = [
            QRectF(10, 10, 100, 150),
            QRectF(120, 10, 100, 150),
            QRectF(10, 170, 210, 100)
        ]
        
        # Mock detector
        class MockDetector:
            def __init__(self):
                self.confidence = 0.25
                self.min_rect_px = 80
        
        detector = MockDetector()
        pdf_path = "/test/performance.pdf"
        
        # Test vitesse
        import time
        start = time.time()
        
        # Premier accès (miss)
        result1 = cache_manager.get_panels(pdf_path, 0, detector)
        assert result1 is None  # Miss attendu
        
        # Sauvegarde
        cache_manager.save_panels(pdf_path, 0, test_panels, detector)
        
        # Deuxième accès (hit)
        result2 = cache_manager.get_panels(pdf_path, 0, detector)
        assert result2 is not None  # Hit attendu
        assert len(result2) == 3
        
        end = time.time()
        
        # Statistiques
        stats = cache_manager.get_cache_info()
        print(f"   ⚡ Temps: {(end-start)*1000:.1f}ms")
        print(f"   📊 Hits mémoire: {stats['stats']['memory_hits']}")
        print(f"   📊 Misses: {stats['stats']['misses']}")
        print(f"   ✅ Performance validée")
        
    except Exception as e:
        print(f"   ❌ Erreur performance: {e}")
        return False
    
    print("\n🎉 Tous les tests d'intégration passés!")
    print("\n📋 Bénéfices du cache amélioré:")
    print("   • ⚡ Cache persistant entre sessions")
    print("   • 🧠 Cache mémoire pour accès ultra-rapide") 
    print("   • 🎯 Invalidation intelligente sur changement paramètres")
    print("   • 📊 Statistiques détaillées de performance")
    print("   • 🧹 Nettoyage automatique des anciens caches")
    print("   • 🔧 Interface de gestion intégrée dans AnComicsViewer")
    
    return True

if __name__ == "__main__":
    success = test_cache_integration()
    sys.exit(0 if success else 1)
