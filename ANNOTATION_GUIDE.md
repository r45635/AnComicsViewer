# Guide d'Annotation Multi-BD 

## ⚠️ IMPORTANT : Workflow Existant
Le projet utilise **scripts/tools/start_annotation.py** pour l'annotation avec Labelme.

**NE PAS CRÉER de nouveaux outils d'annotation sans vérifier l'existant !**

## 📁 Structure Existante
```
dataset/
├── images/train/                         # Images à annoter
│   ├── p0001.png, p0004.png...         # Pages Golden City
│   ├── pinup_p*.png                     # Pages Pin-up (FAIT) 
│   ├── tintin_p*.png                    # Pages Tintin (FAIT)
│   └── sisters_p*.png                   # Pages Sisters (NOUVELLES)
└── labels/train/                         # Annotations Labelme JSON
    ├── p0001.json, p0004.json...       # Annotations Golden City
    ├── pinup_p*.json                    # Annotations Pin-up (FAIT)
    ├── tintin_p*.json                   # Annotations Tintin (FAIT)  
    └── sisters_p*.json                  # Annotations Sisters (TEMPLATES CRÉÉS)
```

## 🚀 Workflow d'Annotation CORRECT

### Utiliser l'outil existant
```bash
source .venv/bin/activate
python scripts/tools/start_annotation.py
```

**Ce script lance automatiquement Labelme avec toutes les images du dataset et préserve les annotations existantes.**

### 3. Types d'annotations à créer

#### 🟦 Classe "panel" (ID: 0)
- **Panneaux principaux** de la BD
- Cases qui contiennent l'action principale
- Tracé : Rectangle autour de chaque panneau

#### 🟨 Classe "panel_inset" (ID: 1) 
- **Encarts** dans les panneaux (bulles, cartouches)
- Éléments textuels flottants
- Zones d'effets spéciaux
- Tracé : Rectangle autour de chaque encart

### 4. Instructions Labelme

1. **Ouvrir image** : `labelme sisters_p012.png`
2. **Créer rectangle** : Clic droit → "Create Rectangle"
3. **Tracer** : Cliquer et glisser pour délimiter le panneau
4. **Nommer** : Taper "panel" ou "panel_inset"
5. **Répéter** pour tous les panneaux/encarts
6. **Sauvegarder** : Ctrl+S → fichier .json créé

### 5. Convertir vers format YOLO
```bash
python scripts/convert_labelme_to_yolo.py \
  --input dataset/annotations_labelme/ \
  --output dataset/labels/train/ \
  --images dataset/images/train/
```

## 📊 Pages à Annoter

1. **sisters_p012.png** - Page 12 (Cassiopeia's Summer)
2. **sisters_p018.png** - Page 18 
3. **sisters_p025.png** - Page 25
4. **sisters_p035.png** - Page 35
5. **sisters_p045.png** - Page 45
6. **sisters_p055.png** - Page 55
7. **sisters_p065.png** - Page 65

## 💡 Conseils d'Annotation

- **Précision** : Bordures ajustées aux vrais contours des panneaux
- **Complétude** : N'oubliez aucun panneau visible
- **Cohérence** : Même critère pour tous les types de panneaux
- **Qualité** : Mieux vaut 5 annotations précises que 10 approximatives

## 🔄 Workflow Complet

```bash
# 1. Annoter avec Labelme
labelme dataset/images/train/sisters_p012.png --output dataset/annotations_labelme/

# 2. Convertir vers YOLO
python scripts/convert_labelme_to_yolo.py \
  --input dataset/annotations_labelme/ \
  --output dataset/labels/train/ \
  --images dataset/images/train/

# 3. Mettre à jour les stats
python scripts/update_dataset_stats.py

# 4. Relancer l'entraînement
python scripts/start_optimized_training.py
```

## 🎯 Objectif Final
- **172 images** au total dans le dataset
- **7 nouvelles pages Sisters** bien annotées
- **Modèle amélioré** pour la détection multi-style (Golden City + Tintin + Pin-up + Sisters)
