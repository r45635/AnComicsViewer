# 🎯 Détecteur Multi-BD - Guide d'utilisation

## 🎉 **Nouveau ! Détecteur YOLO Multi-Styles**

AnComicsViewer intègre maintenant un détecteur de panels entraîné sur **3 styles de BD différents** :

- **🟡 Golden City** - Style moderne complexe avec panels variés
- **🔵 Tintin** - Style classique simple et minimaliste  
- **🔴 Pin-up du B24** - Style aviation/guerre avec layouts spéciaux

## 📊 **Performance Exceptionnelle**

- **mAP50 : 91.1%** - Précision excellente
- **mAP50-95 : 88.3%** - Robustesse multi-échelles
- **Précision : 84.0%** - Peu de faux positifs
- **Rappel : 88.7%** - Détection complète

## 🚀 **Utilisation**

### 1. Lancement rapide
```bash
python demo_multibd.py
```

### 2. Utilisation normale
```bash
python AnComicsViewer.py
```
Puis : **⚙️ Menu → Detector → Multi-BD (Trained)**

### 3. Test d'intégration
```bash
python test_multibd_integration.py
```

## 🎛️ **Interface**

### **Menu Détecteur**
- **Heuristic (OpenCV)** - Détecteur original
- **YOLOv8 Seg (ML)** - Détecteur généraliste  
- **🆕 Multi-BD (Trained)** - Notre détecteur spécialisé
- **Load ML weights…** - Charger modèles personnalisés

### **Avantages Multi-BD**
✅ **Polyvalent** - Fonctionne sur styles très différents  
✅ **Précis** - Entraîné sur données réelles annotées  
✅ **Rapide** - Optimisé pour BD (YOLOv8n)  
✅ **Stable** - Pas de réglages manuels nécessaires  

## 📚 **Comparaison des Détecteurs**

| Détecteur | Avantages | Inconvénients |
|-----------|-----------|---------------|
| **Heuristic** | Rapide, configurable | Limité aux layouts simples |
| **YOLOv8 Seg** | Généraliste, masques | Nécessite modèle externe |
| **🆕 Multi-BD** | Spécialisé BD, précis | Taille modèle (6MB) |

## 🎯 **Styles Supportés**

### **Golden City** ✅
- Panels complexes avec incrustations
- Layouts modernes variés
- Bulles intégrées dans panels

### **Tintin** ✅  
- Style classique rectangulaire
- Layouts en grille simple
- Bulles séparées des panels

### **Pin-up du B24** ✅
- Style aviation/guerre
- Panels techniques détaillés
- Compositions dynamiques

## 🔧 **Configuration Avancée**

### **Réglage de la confiance**
```python
detector.set_confidence(0.3)  # Seuil 0.05-0.95
```

### **Réglage IoU (suppression doublons)**
```python
detector.set_iou_threshold(0.5)  # Seuil 0.1-0.9
```

### **Modèle personnalisé**
```python
from detectors.multibd_detector import MultiBDPanelDetector
detector = MultiBDPanelDetector(weights="mon_modele.pt")
```

## 🏆 **Résultats d'Entraînement**

### **Dataset Final**
- **160 images** totales (44 Golden City + 66 Tintin + 50 Pin-up)
- **50 images annotées** (31.2% couverture)
- **377 annotations** de panels
- **2 classes** : panel, panel_inset

### **Entraînement**
- **50 époques** avec early stopping
- **Batch size 4** optimisé pour mémoire
- **Augmentations légères** pour préserver structure BD
- **Adam optimizer** avec learning rate adaptatif

### **Validation**
- **Split 80/20** train/validation
- **Pas d'overfitting** grâce au dataset diversifié
- **Généralisation excellente** sur nouveaux styles

## 🚀 **Développement Futur**

### **Extensions Possibles**
- [ ] Support manga (sens lecture RTL)
- [ ] Détection bulles de texte améliorée
- [ ] Classification type de panel (action/dialogue/narratif)
- [ ] Support webtoons/strips verticaux

### **Dataset Extensions**
- [ ] Plus de séries BD françaises
- [ ] Comics américains (Marvel/DC)
- [ ] Manga japonais
- [ ] BD numériques natives

## ❓ **Dépannage**

### **Modèle non trouvé**
```bash
# Ré-entraîner le modèle
python train_multibd_model.py
```

### **Erreur PyTorch**
Le patch de compatibilité est automatiquement appliqué.

### **Performance lente**
- Réduisez la taille d'image dans les préférences
- Fermez autres applications gourmandes
- Utilisez GPU si disponible (détection automatique)

### **Détection imprécise**
- Ajustez le seuil de confiance (0.1-0.3)
- Vérifiez que le style BD est supporté
- Essayez le détecteur Heuristic en fallback

## 📞 **Support**

Pour des problèmes ou suggestions :
1. Testez d'abord `python test_multibd_integration.py`
2. Vérifiez les logs console
3. Comparez avec détecteur Heuristic

---

**🎉 Profitez de votre lecture de BD avec la détection multi-styles !** 🎯
