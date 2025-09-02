# AnComicsViewer - Configuration Unifiée Complète

## 🎯 Vue d'Ensemble

Ce fichier `detect.yaml` intègre **toutes les améliorations** développées pour optimiser la détection de panels et ballons dans les comics :

✅ **Tiling intelligent** - Évite le découpage excessif  
✅ **Anti-grille** - Supprime les artefacts de tuilage  
✅ **Fusion adjacente** - Unifie les panels découpés  
✅ **Full-page detection** - Gère les couvertures  
✅ **Debug visuel** - Outils de développement  

---

## 📋 Configuration Complète

### 🎯 **Seuils de Confiance**
```yaml
panel_conf: 0.18              # Seuil minimum pour accepter un panel
balloon_conf: 0.22            # Seuil minimum pour accepter un ballon
```

**Usage :**
- Augmenter → Moins de faux positifs, plus de précision
- Diminuer → Plus de détections, risque de bruit

### 📏 **Contraintes Dimensionnelles**
```yaml
balloon_area_min_pct: 0.0006  # Surface minimale ballon (% page)
balloon_min_w: 30             # Largeur min absolue (px)
balloon_min_h: 22             # Hauteur min absolue (px)
```

**Filtrage :** Supprime les micro-détections parasites

### 🔲 **Tiling Intelligent**
```yaml
imgsz_max: 1536               # Taille max réseau neural
tile_target: 1024             # Taille cible d'une tuile
tile_overlap: 0.25            # Chevauchement (25%)
force_tiling: false           # Auto-détection taille
debug_tiles: false            # Overlay visuel tuiles
```

**Logique :**
- `force_tiling: false` → Pas de tiling si image < 1024×1.2
- `force_tiling: true` → Tiling systématique  
- `debug_tiles: true` → Affiche les tuiles en orange

### ⚙️ **Post-Processing**
```yaml
iou_merge: 0.55                # Fusion IoU standard
panel_containment_merge: 0.65  # Fusion par containment
max_balloons: 15               # Limite nombre ballons
page_margin_inset_pct: 0.015   # Marge page ignorée
max_det: 600                   # Max détections YOLO
```

**Pipeline :** Clamp → IoU Merge → Containment → Size Filter

### 📖 **Full-Page Detection**
```yaml
full_page_panel_pct: 0.80       # Seuil couverture (80%)
full_page_keep_balloons: true   # Garder ballons
full_page_balloon_overlap_pct: 0.15  # Seuil chevauchement
```

**Comportement :**
- Panel ≥ 80% page → Supprime tous les autres panels
- Garde seulement ballons avec ≥ 15% chevauchement

### 🔗 **Fusion Adjacente**
```yaml
panel_merge_iou: 0.3            # Seuil IoU fusion
panel_merge_dist: 0.02          # Distance relative (2%)
```

**Résout :** Bandes blanches découpées, cases adjacentes

---

## 🎛️ **Réglages par Usage**

### 📚 **Comics/BD Européens**
```yaml
panel_conf: 0.20              # Plus strict
panel_merge_iou: 0.25         # Fusion conservative
panel_merge_dist: 0.015       # Cases bien séparées
```

### 📖 **Manga Japonais**
```yaml
panel_conf: 0.15              # Plus permissif
panel_merge_iou: 0.35         # Fusion agressive  
panel_merge_dist: 0.03        # Cases serrées
```

### 🎨 **Webtoons/Webcomics**
```yaml
force_tiling: false           # Souvent verticaux
full_page_panel_pct: 0.90     # Seuil plus strict
panel_merge_dist: 0.04        # Panels très proches
```

### 🔧 **Debug/Développement**
```yaml
debug_tiles: true             # Voir le tiling
max_balloons: 50              # Plus de ballons
panel_conf: 0.10              # Voir plus de détections
```

---

## 📊 **Impact des Paramètres**

### **Résultats Typiques**

| **Paramètre** | **Valeur Basse** | **Valeur Haute** | **Effet** |
|---------------|------------------|------------------|-----------|
| `panel_conf` | 0.10 → 15+ panels | 0.25 → 5 panels | Précision vs Rappel |
| `panel_merge_iou` | 0.2 → Fusion agressive | 0.5 → Fusion conservatrice | Unification |
| `tile_target` | 512 → Plus de tuiles | 1536 → Moins de tuiles | Performance |
| `full_page_panel_pct` | 0.7 → Plus de full-page | 0.9 → Moins de full-page | Couvertures |

### **Performance Attendue**

```
Configuration Défaut:
📄 Page normale  → panels=6-8,  balloons=2-4,  tiles=0-4
📚 Couverture    → panels=1,    balloons=0-2,  no_tiling  
🎨 Planche dense → panels=10-12, balloons=5-8,  tiles=4-9
```

---

## 🚀 **Tests & Validation**

### **Test Standard**
```bash
cd newBranch
python main.py
# Charger un PDF → Observer status bar
```

### **Test CLI**
```bash
cd tools
python eval_one_page.py ../test.pdf 1
```

### **Test Debug**
```yaml
# Dans detect.yaml
debug_tiles: true
panel_conf: 0.10
```

### **Test Performance**
```yaml
# Configuration rapide
imgsz_max: 1024
tile_target: 768
max_det: 300
```

---

## 🔄 **Évolution de la Configuration**

### **Version 1.0** (Basique)
- Seuils fixes, tiling systématique

### **Version 2.0** (Anti-grille)  
- Détection artefacts, profil qualité

### **Version 3.0** (Fusion intelligente)
- Tiling adaptatif, fusion doublons

### **Version 4.0** (Fusion adjacente) 
- Unification panels découpés

### **Version 5.0** (Configuration unifiée) ✅
- Documentation complète, tous paramètres exposés

---

## ✅ **Checklist Validation**

- ✅ **Tous paramètres documentés** - Commentaires explicites
- ✅ **Valeurs optimisées** - Testées sur divers comics  
- ✅ **Compatibilité** - Rétrocompatible avec anciennes versions
- ✅ **Extensibilité** - Facile d'ajouter nouveaux paramètres
- ✅ **Performance** - Équilibre qualité/vitesse

**Configuration prête pour production !** 🚀
