# Guide d'utilisation des modes de détection

## 🎯 Nouveauté : Sélection du mode de détection

Le GUI d'AnComicsViewer propose maintenant **3 modes de détection** pour s'adapter à vos besoins :

### 🤖 **YOLO seul**
- Utilise uniquement le modèle YOLOv8 pour la détection
- Plus rapide, mais peut manquer certains panels
- Idéal pour les bandes dessinées simples

### 📏 **Règles seules**
- Utilise uniquement les algorithmes de traitement d'image
- Détecte les panels par analyse des contours et formes
- Utile pour tester la robustesse des règles

### 🎯 **Hybride (Recommandé)**
- **Combine YOLO + Règles** pour des résultats optimaux
- FUSION INTELLIGENTE des deux approches
- Meilleur rappel et précision
- **Recommandé pour la plupart des utilisations**

## 🔧 Comment utiliser

1. **Lancez** AnComicsViewer
2. **Chargez** un PDF de bande dessinée
3. **Sélectionnez** le mode souhaité dans le menu déroulant de la barre d'outils
4. **Observez** les statistiques dans la console pour voir le nombre de panels détectés

## 📊 Comparaison des résultats

| Mode | Avantages | Inconvénients | Usage recommandé |
|------|-----------|---------------|------------------|
| **YOLO seul** | ⚡ Rapide | ❌ Manque de panels | BD simples |
| **Règles seules** | 🎯 Détection géométrique | ⚠️ Moins robuste | Test/debug |
| **Hybride** | 🏆 Meilleur rappel | 🕐 Plus lent | **Production** |

## 🔍 Dépannage

Si vous voyez moins de panels que prévu :
- ✅ Essayez le mode **Hybride**
- ✅ Vérifiez les paramètres de configuration
- ✅ Activez le debug pour voir les statistiques détaillées

## 💡 Conseil

Le mode **Hybride** est celui qui donne les **meilleurs résultats** car il combine les forces des deux approches :
- YOLO pour la détection basée sur l'apprentissage
- Règles pour la détection géométrique complémentaire</content>
<parameter name="filePath">/Users/vincentcruvellier/Documents/GitHub/AnComicsViewer/Detection_Modes_Guide.md
