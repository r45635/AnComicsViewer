#!/usr/bin/env python3
"""
Cache système avancé pour AnComicsViewer
Optimise les performances avec mise en cache persistante et intelligente
"""

import os
import json
import hashlib
import pickle
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PySide6.QtCore import QRectF, QSizeF, QObject, QThread, Signal
import time

class PanelCacheManager:
    """
    Gestionnaire de cache intelligent pour les détections de panels.
    
    Features:
    - Cache mémoire rapide (session active)
    - Cache disque persistant (entre sessions)
    - Invalidation intelligente (fichier modifié, paramètres changés)
    - Preload en arrière-plan des pages suivantes
    - Compression pour économiser l'espace disque
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or (Path.home() / ".ancomicsviewer" / "cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache mémoire (session courante)
        self._memory_cache: Dict[str, Dict[int, List[QRectF]]] = {}
        
        # Métadonnées des fichiers
        self._file_metadata: Dict[str, dict] = {}
        
        # Cache des hashs de paramètres détecteur
        self._detector_hashes: Dict[str, str] = {}
        
        # Statistiques
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0, 
            "misses": 0,
            "saves": 0
        }
        
        # Lock pour thread safety
        self._lock = threading.RLock()
        
        print(f"📁 Cache initialisé: {self.cache_dir}")
    
    def _get_file_hash(self, pdf_path: str) -> str:
        """Calcule un hash unique pour le fichier PDF."""
        try:
            stat = os.stat(pdf_path)
            content = f"{pdf_path}:{stat.st_size}:{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except:
            return hashlib.md5(pdf_path.encode()).hexdigest()[:16]
    
    def _get_detector_hash(self, detector) -> str:
        """Génère un hash basé sur les paramètres du détecteur."""
        try:
            # VERSION DE CACHE - v5 avec panels-by-name
            CACHE_VERSION = "v5-panels-by-name"
            
            # Récupère les paramètres principaux du détecteur
            params = {
                "cache_version": CACHE_VERSION,  # ← Force la régénération du cache
                "device": getattr(detector, 'device', 'cpu'),
                "model_name": getattr(detector, 'model_name', 'unknown')
            }
            
            # Ajoute la config BD si disponible
            if hasattr(detector, 'config'):
                config = detector.config
                params.update({
                    "conf_base": getattr(config, 'CONF_BASE', 0.15),
                    "conf_min": getattr(config, 'CONF_MIN', 0.05),
                    "iou_nms": getattr(config, 'IOU_NMS', 0.5),
                    "target_min": getattr(config, 'TARGET_MIN', 2),
                    "target_max": getattr(config, 'TARGET_MAX', 20),
                })
            
            # Ajoute la signature des noms de classes du modèle si disponible
            if hasattr(detector, 'get_model_names_signature'):
                try:
                    params["model_names_sig"] = detector.get_model_names_signature()
                except:
                    pass
                
            # Ajoute tous les autres paramètres publics (fallback)
            for key, value in detector.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    try:
                        params[key] = str(value)
                    except:
                        continue
            
            param_str = json.dumps(params, sort_keys=True)
            return hashlib.md5(param_str.encode()).hexdigest()[:12]
        except:
            pass
        return "default_v4"
    
    def _get_cache_key(self, pdf_path: str, detector) -> Tuple[str, str]:
        """Génère les clés de cache pour un fichier et détecteur."""
        file_hash = self._get_file_hash(pdf_path)
        detector_hash = self._get_detector_hash(detector)
        return file_hash, detector_hash
    
    def _get_cache_file(self, pdf_path: str, detector) -> Path:
        """Retourne le chemin du fichier de cache."""
        file_hash, detector_hash = self._get_cache_key(pdf_path, detector)
        cache_filename = f"{file_hash}_{detector_hash}.cache"
        return self.cache_dir / cache_filename
    
    def get_panels(self, pdf_path: str, page: int, detector) -> Optional[List[QRectF]]:
        """
        Récupère les panels depuis le cache (mémoire puis disque).
        
        Returns:
            List[QRectF] si trouvé dans le cache, None sinon
        """
        with self._lock:
            cache_key = self._get_cache_key(pdf_path, detector)[0]
            
            # 1. Vérifier le cache mémoire
            if cache_key in self._memory_cache:
                if page in self._memory_cache[cache_key]:
                    self.stats["memory_hits"] += 1
                    return self._memory_cache[cache_key][page]
            
            # 2. Vérifier le cache disque
            cache_file = self._get_cache_file(pdf_path, detector)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if page in cache_data.get('pages', {}):
                        panels_data = cache_data['pages'][page]
                        
                        # Reconstruire les QRectF
                        panels = []
                        for rect_data in panels_data:
                            rect = QRectF(rect_data['x'], rect_data['y'], 
                                        rect_data['w'], rect_data['h'])
                            panels.append(rect)
                        
                        # Mettre en cache mémoire pour la prochaine fois
                        if cache_key not in self._memory_cache:
                            self._memory_cache[cache_key] = {}
                        self._memory_cache[cache_key][page] = panels
                        
                        self.stats["disk_hits"] += 1
                        return panels
                        
                except Exception as e:
                    print(f"⚠️  Erreur lecture cache {cache_file}: {e}")
            
            self.stats["misses"] += 1
            return None
    
    def save_panels(self, pdf_path: str, page: int, panels: List[QRectF], detector):
        """
        Sauvegarde les panels dans le cache (mémoire + disque).
        ⚠️ NE SAUVEGARDE PAS les résultats vides pour laisser une 2e chance.
        """
        with self._lock:
            # Ne pas sauvegarder les résultats vides dans le cache
            if not panels or len(panels) == 0:
                print(f"[Cache] Skipping empty result for page {page} (no cache write)")
                return
            
            cache_key = self._get_cache_key(pdf_path, detector)[0]
            
            # 1. Cache mémoire
            if cache_key not in self._memory_cache:
                self._memory_cache[cache_key] = {}
            self._memory_cache[cache_key][page] = panels
            
            # 2. Cache disque
            cache_file = self._get_cache_file(pdf_path, detector)
            
            try:
                # Charger les données existantes ou créer nouvelles
                cache_data = {'pages': {}, 'metadata': {}}
                if cache_file.exists():
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                
                # Convertir QRectF en données sérialisables
                panels_data = []
                for rect in panels:
                    panels_data.append({
                        'x': rect.x(),
                        'y': rect.y(), 
                        'w': rect.width(),
                        'h': rect.height()
                    })
                
                cache_data['pages'][page] = panels_data
                cache_data['metadata'] = {
                    'timestamp': time.time(),
                    'detector_type': type(detector).__name__,
                    'panel_count': len(panels)
                }
                
                # Sauvegarder
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                self.stats["saves"] += 1
                
            except Exception as e:
                print(f"⚠️  Erreur sauvegarde cache {cache_file}: {e}")
    
    def clear_file_cache(self, pdf_path: str):
        """Efface le cache pour un fichier PDF spécifique."""
        with self._lock:
            file_hash = self._get_file_hash(pdf_path)
            
            # Cache mémoire
            if file_hash in self._memory_cache:
                del self._memory_cache[file_hash]
            
            # Cache disque - supprimer tous les fichiers avec ce hash
            pattern = f"{file_hash}_*.cache"
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                    print(f"🗑️  Cache supprimé: {cache_file.name}")
                except Exception as e:
                    print(f"⚠️  Erreur suppression {cache_file}: {e}")
    
    def clear_all_cache(self):
        """Efface tout le cache."""
        with self._lock:
            # Cache mémoire
            self._memory_cache.clear()
            
            # Cache disque
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    print(f"⚠️  Erreur suppression {cache_file}: {e}")
            
            print("🧹 Cache complètement vidé")
    
    def get_cache_info(self) -> dict:
        """Retourne les informations sur le cache."""
        with self._lock:
            memory_files = len(self._memory_cache)
            memory_pages = sum(len(pages) for pages in self._memory_cache.values())
            
            disk_files = len(list(self.cache_dir.glob("*.cache")))
            disk_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.cache"))
            
            return {
                "memory": {
                    "files": memory_files,
                    "pages": memory_pages
                },
                "disk": {
                    "files": disk_files,
                    "size_mb": disk_size / (1024 * 1024),
                    "path": str(self.cache_dir)
                },
                "stats": self.stats.copy()
            }
    
    def cleanup_old_cache(self, max_age_days: int = 30):
        """Nettoie les fichiers de cache anciens."""
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        cleaned = 0
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    cleaned += 1
            except Exception as e:
                print(f"⚠️  Erreur nettoyage {cache_file}: {e}")
        
        if cleaned > 0:
            print(f"🧹 {cleaned} fichiers de cache anciens supprimés")


class PreloadWorker(QThread):
    """
    Worker thread pour preload des pages suivantes en arrière-plan.
    """
    
    preload_completed = Signal(int, list)  # page, panels
    
    def __init__(self, pdf_path: str, pages_to_load: List[int], detector, document):
        super().__init__()
        self.pdf_path = pdf_path
        self.pages_to_load = pages_to_load
        self.detector = detector
        self.document = document
        self._stop_requested = False
    
    def stop(self):
        self._stop_requested = True
    
    def run(self):
        """Précharge les pages en arrière-plan."""
        for page in self.pages_to_load:
            if self._stop_requested:
                break
            
            try:
                # Simuler la détection (remplacer par la vraie logique)
                pt = self.document.pagePointSize(page)
                dpi = 200  # DPI par défaut
                scale = dpi / 72.0
                qsize = QSizeF(pt.width() * scale, pt.height() * scale).toSize()
                qimg = self.document.render(page, qsize)
                
                panels = self.detector.detect_panels(qimg, pt)
                
                self.preload_completed.emit(page, panels)
                
                # Petite pause pour ne pas surcharger
                self.msleep(100)
                
            except Exception as e:
                print(f"⚠️  Erreur preload page {page}: {e}")


def create_enhanced_cache_test():
    """Crée un test pour le système de cache avancé."""
    print("🧪 Test du Cache Manager Enhanced")
    print("=" * 40)
    
    # Créer le cache manager
    cache_manager = PanelCacheManager()
    
    # Simuler un détecteur simple
    class MockDetector:
        def __init__(self):
            self.adaptive_block = 51
            self.min_rect_px = 80
            self.confidence = 0.25
    
    detector = MockDetector()
    pdf_path = "/test/comics.pdf"
    
    # Test 1: Cache miss
    print("1. Test cache miss...")
    panels = cache_manager.get_panels(pdf_path, 0, detector)
    assert panels is None
    print("   ✅ Cache miss détecté")
    
    # Test 2: Sauvegarde
    print("2. Test sauvegarde...")
    test_panels = [
        QRectF(10, 10, 100, 150),
        QRectF(120, 10, 100, 150),
        QRectF(10, 170, 210, 100)
    ]
    cache_manager.save_panels(pdf_path, 0, test_panels, detector)
    print("   ✅ Panels sauvegardés")
    
    # Test 3: Cache hit mémoire
    print("3. Test cache hit mémoire...")
    cached_panels = cache_manager.get_panels(pdf_path, 0, detector)
    assert cached_panels is not None
    assert len(cached_panels) == 3
    print(f"   ✅ {len(cached_panels)} panels récupérés de la mémoire")
    
    # Test 4: Cache hit disque (simuler redémarrage)
    print("4. Test cache hit disque...")
    cache_manager._memory_cache.clear()  # Vider la mémoire
    cached_panels = cache_manager.get_panels(pdf_path, 0, detector)
    assert cached_panels is not None
    assert len(cached_panels) == 3
    print(f"   ✅ {len(cached_panels)} panels récupérés du disque")
    
    # Test 5: Informations cache
    print("5. Test informations cache...")
    info = cache_manager.get_cache_info()
    print(f"   📊 Mémoire: {info['memory']['files']} fichiers, {info['memory']['pages']} pages")
    print(f"   💾 Disque: {info['disk']['files']} fichiers, {info['disk']['size_mb']:.1f} MB")
    print(f"   📈 Stats: {info['stats']}")
    
    # Test 6: Nettoyage
    print("6. Test nettoyage...")
    cache_manager.clear_file_cache(pdf_path)
    panels = cache_manager.get_panels(pdf_path, 0, detector)
    assert panels is None
    print("   ✅ Cache nettoyé")
    
    print()
    print("🎉 Tous les tests du cache passés!")
    
    return cache_manager

if __name__ == "__main__":
    create_enhanced_cache_test()
