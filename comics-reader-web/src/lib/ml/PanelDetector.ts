/**
 * Détecteur de panels embarqué pour le web utilisant ONNX Runtime Web
 * Adapté du modèle Multi-BD YOLOv8 finetuné
 */

import { InferenceSession, Tensor } from 'onnxruntime-web';
import { Panel } from '@/types';

interface DetectorConfig {
  confidenceThreshold: number;
  iouThreshold: number;
  maxDetections: number;
  inputSize: number;
}

interface ModelInfo {
  performance: {
    mAP50: number;
  };
  classes: string[];
  version: string;
}

interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  classId: number;
  className: string;
}

export class EmbeddedPanelDetector {
  private session: InferenceSession | null = null;
  private isInitialized = false;
  private modelInfo: ModelInfo | null = null;
  
  private config: DetectorConfig = {
    confidenceThreshold: 0.2,
    iouThreshold: 0.5,
    maxDetections: 100,
    inputSize: 640
  };

  constructor() {
    // Configuration du Web Worker si nécessaire
    if (typeof window !== 'undefined') {
      // Configuration pour le navigateur
      this.setupWebRuntime();
    }
  }

  /**
   * Configure ONNX Runtime pour le web
   */
  private setupWebRuntime() {
    // Optimisations pour le navigateur
    if (typeof window !== 'undefined') {
      // Configuration dynamique pour éviter le require()
      import('onnxruntime-web').then((ort) => {
        ort.env.wasm.wasmPaths = '/models/';
        ort.env.logLevel = 'error';
      }).catch(() => {
        console.warn('⚠️ Impossible de configurer ONNX Runtime');
      });
    }
  }

  /**
   * Initialise le détecteur avec le modèle ONNX
   */
  async initialize(): Promise<void> {
    try {
      console.log('🔄 Initialisation du détecteur Multi-BD web...');
      
      // Charger les informations du modèle
      try {
        const infoResponse = await fetch('/models/model_info.json');
        if (infoResponse.ok) {
          this.modelInfo = await infoResponse.json();
        }
      } catch {
        console.warn('⚠️ Impossible de charger model_info.json');
      }
      
      // Créer la session ONNX avec optimisations web
      this.session = await InferenceSession.create('/models/multibd_model.onnx', {
        executionProviders: [
          'webgl',  // GPU si disponible
          'wasm'    // Fallback CPU
        ],
        enableCpuMemArena: true,
        enableMemPattern: true,
        executionMode: 'sequential',
        logSeverityLevel: 3
      });
      
      this.isInitialized = true;
      console.log('✅ Détecteur Multi-BD web initialisé');
      
      if (this.modelInfo) {
        console.log(`📊 Performance: ${this.modelInfo.performance.mAP50} mAP50`);
        console.log(`🎯 Classes: ${this.modelInfo.classes.join(', ')}`);
      }
      
    } catch (error) {
      console.error('❌ Erreur initialisation détecteur:', error);
      throw error;
    }
  }

  /**
   * Détecte les panels dans une image
   */
  async detectPanels(imageUri: string): Promise<Panel[]> {
    try {
      if (!this.isInitialized) {
        await this.initialize();
      }

      console.log('🔍 Détection panels en cours...');
      const startTime = Date.now();
      
      // 1. Preprocessing de l'image
      const inputTensor = await this.preprocessImage(imageUri);
      
      // 2. Inférence ONNX
      const results = await this.session!.run({
        images: inputTensor
      });
      
      // 3. Postprocessing YOLO
      const detections = this.parseYOLOOutput(results.output0 || results.output);
      
      // 4. Convertir en format Panel avec coordonnées relatives
      const panels = this.convertToPanels(detections);
      
      const inferenceTime = Date.now() - startTime;
      console.log(`⚡ Détection terminée: ${panels.length} panels en ${inferenceTime}ms`);
      
      return panels;
      
    } catch (error) {
      console.error('❌ Erreur détection panels, utilisation heuristique de fallback:', error);
      return this.detectPanelsWithHeuristic(imageUri);
    }
  }

  /**
   * Preprocessing d'image pour YOLOv8 (web)
   */
  private async preprocessImage(imageUri: string): Promise<Tensor> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          // Créer un canvas pour le preprocessing
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            throw new Error('Impossible de créer le contexte canvas');
          }
          
          // Redimensionner à 640x640
          canvas.width = this.config.inputSize;
          canvas.height = this.config.inputSize;
          
          // Dessiner l'image redimensionnée (letterbox pour préserver l'aspect ratio)
          const scale = Math.min(
            this.config.inputSize / img.width,
            this.config.inputSize / img.height
          );
          
          const scaledWidth = img.width * scale;
          const scaledHeight = img.height * scale;
          const x = (this.config.inputSize - scaledWidth) / 2;
          const y = (this.config.inputSize - scaledHeight) / 2;
          
          // Fond noir pour le letterboxing
          ctx.fillStyle = '#000000';
          ctx.fillRect(0, 0, this.config.inputSize, this.config.inputSize);
          
          // Dessiner l'image centrée
          ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
          
          // Extraire les données pixels
          const imageData = ctx.getImageData(0, 0, this.config.inputSize, this.config.inputSize);
          const pixels = imageData.data;
          
          // Convertir RGBA vers RGB normalisé [0,1] format CHW
          const inputArray = new Float32Array(3 * this.config.inputSize * this.config.inputSize);
          const pixelCount = this.config.inputSize * this.config.inputSize;
          
          for (let i = 0; i < pixelCount; i++) {
            const pixelIdx = i * 4; // RGBA
            
            // Normalisation RGB [0,255] -> [0,1] et réorganisation CHW
            inputArray[i] = pixels[pixelIdx] / 255.0;                    // R channel
            inputArray[i + pixelCount] = pixels[pixelIdx + 1] / 255.0;   // G channel
            inputArray[i + 2 * pixelCount] = pixels[pixelIdx + 2] / 255.0; // B channel
          }
          
          // Créer le tensor ONNX [1, 3, 640, 640]
          const tensor = new Tensor('float32', inputArray, [1, 3, this.config.inputSize, this.config.inputSize]);
          resolve(tensor);
          
        } catch (error) {
          reject(error);
        }
      };
      
      img.onerror = () => reject(new Error('Impossible de charger l\'image'));
      img.src = imageUri;
    });
  }

  /**
   * Parse la sortie YOLO et applique NMS
   */
  private parseYOLOOutput(output: Tensor): Detection[] {
    try {
      const outputData = output.data as Float32Array;
      
      // YOLOv8 output format: [1, 84, 8400] ou [1, num_boxes, 84]
      // 84 = 4 (bbox) + 80 (classes) pour COCO, mais ici on a moins de classes
      const shape = output.dims;
      
      let numBoxes: number;
      let numFeatures: number;
      
      if (shape.length === 3) {
        if (shape[1] > shape[2]) {
          // Format [1, num_features, num_boxes]
          numFeatures = shape[1];
          numBoxes = shape[2];
        } else {
          // Format [1, num_boxes, num_features]
          numBoxes = shape[1];
          numFeatures = shape[2];
        }
      } else {
        throw new Error(`Format de sortie YOLO non supporté: ${shape}`);
      }
      
      const detections: Detection[] = [];
      
      // Traitement selon le format
      if (numFeatures > numBoxes) {
        // Format transposé [1, features, boxes]
        for (let i = 0; i < numBoxes; i++) {
          const x = outputData[i];
          const y = outputData[numBoxes + i];
          const w = outputData[2 * numBoxes + i];
          const h = outputData[3 * numBoxes + i];
          
          // Les scores de classe commencent à l'indice 4 * numBoxes
          let maxConf = 0;
          let maxClassId = 0;
          
          for (let classIdx = 0; classIdx < numFeatures - 4; classIdx++) {
            const conf = outputData[(4 + classIdx) * numBoxes + i];
            if (conf > maxConf) {
              maxConf = conf;
              maxClassId = classIdx;
            }
          }
          
          if (maxConf >= this.config.confidenceThreshold) {
            detections.push({
              x: (x - w / 2) / this.config.inputSize,  // Normaliser et convertir center -> top-left
              y: (y - h / 2) / this.config.inputSize,
              width: w / this.config.inputSize,
              height: h / this.config.inputSize,
              confidence: maxConf,
              classId: maxClassId,
              className: this.getClassName(maxClassId)
            });
          }
        }
      } else {
        // Format normal [1, boxes, features]
        for (let i = 0; i < numBoxes; i++) {
          const startIdx = i * numFeatures;
          
          const x = outputData[startIdx];
          const y = outputData[startIdx + 1];
          const w = outputData[startIdx + 2];
          const h = outputData[startIdx + 3];
          
          // Trouver la classe avec la plus haute confiance
          let maxConf = 0;
          let maxClassId = 0;
          
          for (let classIdx = 0; classIdx < numFeatures - 4; classIdx++) {
            const conf = outputData[startIdx + 4 + classIdx];
            if (conf > maxConf) {
              maxConf = conf;
              maxClassId = classIdx;
            }
          }
          
          if (maxConf >= this.config.confidenceThreshold) {
            detections.push({
              x: (x - w / 2) / this.config.inputSize,
              y: (y - h / 2) / this.config.inputSize,
              width: w / this.config.inputSize,
              height: h / this.config.inputSize,
              confidence: maxConf,
              classId: maxClassId,
              className: this.getClassName(maxClassId)
            });
          }
        }
      }
      
      // Appliquer NMS
      return this.applyNMS(detections);
      
    } catch (error) {
      console.error('❌ Erreur parsing YOLO:', error);
      return [];
    }
  }

  /**
   * Applique Non-Maximum Suppression
   */
  private applyNMS(detections: Detection[]): Detection[] {
    // Trier par confiance décroissante
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const filtered: Detection[] = [];
    
    for (const detection of detections) {
      let keep = true;
      
      // Vérifier le chevauchement avec les détections déjà gardées
      for (const kept of filtered) {
        const iou = this.calculateIoU(detection, kept);
        if (iou > this.config.iouThreshold) {
          keep = false;
          break;
        }
      }
      
      if (keep) {
        filtered.push(detection);
      }
      
      // Limiter le nombre de détections
      if (filtered.length >= this.config.maxDetections) {
        break;
      }
    }
    
    return filtered;
  }

  /**
   * Calcule l'Intersection over Union entre deux boîtes
   */
  private calculateIoU(box1: Detection, box2: Detection): number {
    const x1 = Math.max(box1.x, box2.x);
    const y1 = Math.max(box1.y, box2.y);
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width);
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height);
    
    if (x2 <= x1 || y2 <= y1) {
      return 0;
    }
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = box1.width * box1.height;
    const area2 = box2.width * box2.height;
    const union = area1 + area2 - intersection;
    
    return intersection / union;
  }

  /**
   * Convertit les détections en panels avec coordonnées relatives
   */
  private convertToPanels(detections: Detection[]): Panel[] {
    return detections.map((detection, index) => ({
      id: index,
      x: Math.max(0, Math.min(1, detection.x)),
      y: Math.max(0, Math.min(1, detection.y)),
      width: Math.max(0, Math.min(1, detection.width)),
      height: Math.max(0, Math.min(1, detection.height)),
      confidence: detection.confidence
    }));
  }

  /**
   * Récupère le nom de classe depuis l'ID
   */
  private getClassName(classId: number): string {
    const classNames = ['panel', 'panel_inset'];
    return classNames[classId] || 'unknown';
  }

  /**
   * Configure les paramètres de détection
   */
  setConfig(config: Partial<DetectorConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('⚙️ Configuration mise à jour:', this.config);
  }

  /**
   * Récupère les informations du modèle
   */
  getModelInfo() {
    return {
      isInitialized: this.isInitialized,
      config: this.config,
      modelInfo: this.modelInfo
    };
  }

  /**
   * Détection heuristique de fallback quand le modèle ONNX n'est pas disponible
   */
  private async detectPanelsWithHeuristic(imageUri: string): Promise<Panel[]> {
    console.log('🎯 Utilisation de l\'heuristique avancée pour la détection de panels');
    
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      
      img.onload = () => {
        try {
          // Créer un canvas pour analyser l'image
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          
          if (!ctx) {
            resolve(this.generateHeuristicPanels(img.width, img.height));
            return;
          }
          
          // Redimensionner pour l'analyse (plus rapide)
          const analysisWidth = Math.min(800, img.width);
          const analysisHeight = Math.floor(img.height * (analysisWidth / img.width));
          
          canvas.width = analysisWidth;
          canvas.height = analysisHeight;
          ctx.drawImage(img, 0, 0, analysisWidth, analysisHeight);
          
          // Analyser l'image pour détecter les panels
          const panels = this.analyzeImageForPanels(ctx, analysisWidth, analysisHeight, img.width, img.height);
          console.log(`📐 Heuristique avancée: ${panels.length} panels détectés`);
          resolve(panels);
          
        } catch (error) {
          console.error('❌ Erreur heuristique avancée, fallback vers grille:', error);
          resolve(this.generateHeuristicPanels(img.width, img.height));
        }
      };
      
      img.onerror = () => {
        console.error('❌ Erreur chargement image pour heuristique');
        resolve([]);
      };
      
      img.src = imageUri;
    });
  }

  /**
   * Analyse une image pour détecter les panels en utilisant des techniques de vision par ordinateur
   */
  private analyzeImageForPanels(
    ctx: CanvasRenderingContext2D, 
    width: number, 
    height: number, 
    originalWidth: number, 
    originalHeight: number
  ): Panel[] {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // 1. Détecter les lignes de séparation horizontales et verticales
    const horizontalLines = this.detectSeparatorLines(data, width, height, 'horizontal');
    const verticalLines = this.detectSeparatorLines(data, width, height, 'vertical');
    
    // 2. Créer une grille basée sur les lignes détectées
    const panels = this.createPanelsFromLines(horizontalLines, verticalLines, width, height);
    
    // 3. Si peu de panels détectés, fallback vers analyse de densité
    if (panels.length < 2) {
      return this.detectPanelsByDensity(data, width, height, originalWidth, originalHeight);
    }
    
    // 4. Convertir vers les coordonnées originales
    const scaleX = originalWidth / width;
    const scaleY = originalHeight / height;
    
    return panels.map((panel, index) => ({
      id: index,
      x: panel.x / width, // Déjà en coordonnées relatives [0,1]
      y: panel.y / height,
      width: panel.width / width,
      height: panel.height / height,
      confidence: panel.confidence
    }));
  }

  /**
   * Détecte les lignes de séparation entre panels
   */
  private detectSeparatorLines(
    data: Uint8ClampedArray, 
    width: number, 
    height: number, 
    direction: 'horizontal' | 'vertical'
  ): number[] {
    const lines: number[] = [];
    const threshold = 0.8; // Seuil pour considérer une ligne comme séparatrice
    const minLineLength = direction === 'horizontal' ? width * 0.6 : height * 0.6;
    
    if (direction === 'horizontal') {
      // Analyser chaque ligne horizontale
      for (let y = 10; y < height - 10; y += 5) {
        let whitePixels = 0;
        let totalPixels = 0;
        
        for (let x = 10; x < width - 10; x += 2) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          
          // Détecter les pixels blancs/clairs (gouttières)
          const brightness = (r + g + b) / 3;
          if (brightness > 240 || (Math.abs(r - g) < 10 && Math.abs(g - b) < 10 && brightness > 200)) {
            whitePixels++;
          }
          totalPixels++;
        }
        
        // Si la ligne est majoritairement blanche/claire, c'est une séparation
        if (totalPixels > minLineLength && whitePixels / totalPixels > threshold) {
          lines.push(y);
        }
      }
    } else {
      // Analyser chaque ligne verticale
      for (let x = 10; x < width - 10; x += 5) {
        let whitePixels = 0;
        let totalPixels = 0;
        
        for (let y = 10; y < height - 10; y += 2) {
          const idx = (y * width + x) * 4;
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          
          const brightness = (r + g + b) / 3;
          if (brightness > 240 || (Math.abs(r - g) < 10 && Math.abs(g - b) < 10 && brightness > 200)) {
            whitePixels++;
          }
          totalPixels++;
        }
        
        if (totalPixels > minLineLength && whitePixels / totalPixels > threshold) {
          lines.push(x);
        }
      }
    }
    
    // Fusionner les lignes proches
    return this.mergeCloseLines(lines, direction === 'horizontal' ? height : width);
  }

  /**
   * Fusionne les lignes de séparation proches
   */
  private mergeCloseLines(lines: number[], imageSize: number): number[] {
    if (lines.length === 0) return lines;
    
    const mergedLines: number[] = [];
    const minDistance = imageSize * 0.05; // 5% de la taille de l'image
    
    lines.sort((a, b) => a - b);
    
    let currentGroup = [lines[0]];
    
    for (let i = 1; i < lines.length; i++) {
      if (lines[i] - lines[i-1] < minDistance) {
        currentGroup.push(lines[i]);
      } else {
        // Moyenne du groupe actuel
        const avgLine = currentGroup.reduce((sum, line) => sum + line, 0) / currentGroup.length;
        mergedLines.push(Math.round(avgLine));
        currentGroup = [lines[i]];
      }
    }
    
    // Ajouter le dernier groupe
    const avgLine = currentGroup.reduce((sum, line) => sum + line, 0) / currentGroup.length;
    mergedLines.push(Math.round(avgLine));
    
    return mergedLines;
  }

  /**
   * Crée des panels à partir des lignes de séparation détectées
   */
  private createPanelsFromLines(
    horizontalLines: number[], 
    verticalLines: number[], 
    width: number, 
    height: number
  ): Panel[] {
    const panels: Panel[] = [];
    
    // Ajouter les bordures comme lignes de séparation
    const hLines = [0, ...horizontalLines, height].sort((a, b) => a - b);
    const vLines = [0, ...verticalLines, width].sort((a, b) => a - b);
    
    // Créer des panels pour chaque cellule de la grille
    for (let i = 0; i < hLines.length - 1; i++) {
      for (let j = 0; j < vLines.length - 1; j++) {
        const x = vLines[j];
        const y = hLines[i];
        const w = vLines[j + 1] - vLines[j];
        const h = hLines[i + 1] - hLines[i];
        
        // Ignorer les panels trop petits
        if (w > width * 0.1 && h > height * 0.1) {
          panels.push({
            id: panels.length,
            x: x + w * 0.02, // Petite marge
            y: y + h * 0.02,
            width: w * 0.96,
            height: h * 0.96,
            confidence: 0.8
          });
        }
      }
    }
    
    return panels;
  }

  /**
   * Détection basée sur l'analyse de densité des couleurs
   */
  private detectPanelsByDensity(
    data: Uint8ClampedArray, 
    width: number, 
    height: number, 
    originalWidth: number, 
    originalHeight: number
  ): Panel[] {
    // Diviser l'image en régions et analyser la variance des couleurs
    const gridSize = 8;
    const cellWidth = Math.floor(width / gridSize);
    const cellHeight = Math.floor(height / gridSize);
    const densityMap: number[][] = [];
    
    // Calculer la densité de chaque cellule
    for (let row = 0; row < gridSize; row++) {
      densityMap[row] = [];
      for (let col = 0; col < gridSize; col++) {
        const startX = col * cellWidth;
        const startY = row * cellHeight;
        const density = this.calculateRegionDensity(data, width, startX, startY, cellWidth, cellHeight);
        densityMap[row][col] = density;
      }
    }
    
    // Regrouper les cellules avec densité similaire en panels
    const panels = this.groupDensityRegions(densityMap, gridSize, cellWidth, cellHeight, originalWidth, originalHeight);
    
    // Si aucun panel détecté, utiliser la grille de fallback
    return panels.length > 0 ? panels : this.generateHeuristicPanels(originalWidth, originalHeight);
  }

  /**
   * Calcule la densité (variance) d'une région
   */
  private calculateRegionDensity(
    data: Uint8ClampedArray, 
    width: number, 
    startX: number, 
    startY: number, 
    regionWidth: number, 
    regionHeight: number
  ): number {
    let totalVariance = 0;
    let pixelCount = 0;
    
    for (let y = startY; y < startY + regionHeight && y < data.length / (width * 4); y += 2) {
      for (let x = startX; x < startX + regionWidth && x < width; x += 2) {
        const idx = (y * width + x) * 4;
        if (idx + 3 < data.length) {
          const r = data[idx];
          const g = data[idx + 1];
          const b = data[idx + 2];
          
          // Calculer la variance RGB
          const mean = (r + g + b) / 3;
          const variance = ((r - mean) ** 2 + (g - mean) ** 2 + (b - mean) ** 2) / 3;
          totalVariance += variance;
          pixelCount++;
        }
      }
    }
    
    return pixelCount > 0 ? totalVariance / pixelCount : 0;
  }

  /**
   * Regroupe les régions de densité similaire en panels
   */
  private groupDensityRegions(
    densityMap: number[][], 
    gridSize: number, 
    cellWidth: number, 
    cellHeight: number, 
    originalWidth: number, 
    originalHeight: number
  ): Panel[] {
    const panels: Panel[] = [];
    const visited: boolean[][] = Array(gridSize).fill(null).map(() => Array(gridSize).fill(false));
    
    // Seuil de similarité pour regrouper les cellules
    const similarityThreshold = 50;
    
    for (let row = 0; row < gridSize; row++) {
      for (let col = 0; col < gridSize; col++) {
        if (!visited[row][col] && densityMap[row][col] > 10) { // Ignorer les régions trop uniformes
          const region = this.floodFillDensity(densityMap, visited, row, col, gridSize, similarityThreshold);
          
          if (region.length > 1) { // Au moins 2 cellules
            const panel = this.regionToBoundingBox(region, cellWidth, cellHeight, originalWidth, originalHeight);
            panels.push(panel);
          }
        }
      }
    }
    
    return panels;
  }

  /**
   * Algorithme de flood fill pour regrouper les cellules de densité similaire
   */
  private floodFillDensity(
    densityMap: number[][], 
    visited: boolean[][], 
    startRow: number, 
    startCol: number, 
    gridSize: number, 
    threshold: number
  ): Array<{row: number, col: number}> {
    const region: Array<{row: number, col: number}> = [];
    const stack = [{row: startRow, col: startCol}];
    const referenceDensity = densityMap[startRow][startCol];
    
    while (stack.length > 0) {
      const {row, col} = stack.pop()!;
      
      if (row < 0 || row >= gridSize || col < 0 || col >= gridSize || visited[row][col]) {
        continue;
      }
      
      if (Math.abs(densityMap[row][col] - referenceDensity) > threshold) {
        continue;
      }
      
      visited[row][col] = true;
      region.push({row, col});
      
      // Ajouter les voisins
      stack.push({row: row - 1, col});
      stack.push({row: row + 1, col});
      stack.push({row, col: col - 1});
      stack.push({row, col: col + 1});
    }
    
    return region;
  }

  /**
   * Convertit une région en bounding box
   */
  private regionToBoundingBox(
    region: Array<{row: number, col: number}>, 
    cellWidth: number, 
    cellHeight: number, 
    originalWidth: number, 
    originalHeight: number
  ): Panel {
    const minRow = Math.min(...region.map(r => r.row));
    const maxRow = Math.max(...region.map(r => r.row));
    const minCol = Math.min(...region.map(r => r.col));
    const maxCol = Math.max(...region.map(r => r.col));
    
    const x = minCol * cellWidth;
    const y = minRow * cellHeight;
    const width = (maxCol - minCol + 1) * cellWidth;
    const height = (maxRow - minRow + 1) * cellHeight;
    
    return {
      id: 0, // Sera réassigné plus tard
      x: x / originalWidth,
      y: y / originalHeight,
      width: width / originalWidth,
      height: height / originalHeight,
      confidence: 0.75
    };
  }

  /**
   * Génère des panels en utilisant une grille heuristique basée sur les formats BD courants
   */
  private generateHeuristicPanels(imageWidth: number, imageHeight: number): Panel[] {
    const panels: Panel[] = [];
    
    // Analyser le ratio de l'image pour déterminer le layout
    const aspectRatio = imageWidth / imageHeight;
    
    if (aspectRatio > 1.4) {
      // Format paysage - probablement 2 pages côte à côte
      // Diviser en 2 colonnes avec quelques panels par colonne
      const colWidth = 0.48; // 48% de largeur par colonne
      const margins = 0.02;
      
      // Colonne gauche
      panels.push(
        { id: 0, x: margins, y: 0.05, width: colWidth, height: 0.25, confidence: 0.8 },
        { id: 1, x: margins, y: 0.32, width: colWidth, height: 0.35, confidence: 0.8 },
        { id: 2, x: margins, y: 0.70, width: colWidth, height: 0.25, confidence: 0.8 }
      );
      
      // Colonne droite
      panels.push(
        { id: 3, x: 0.5 + margins, y: 0.05, width: colWidth, height: 0.25, confidence: 0.8 },
        { id: 4, x: 0.5 + margins, y: 0.32, width: colWidth, height: 0.35, confidence: 0.8 },
        { id: 5, x: 0.5 + margins, y: 0.70, width: colWidth, height: 0.25, confidence: 0.8 }
      );
      
    } else {
      // Format portrait - une page normale
      if (aspectRatio > 0.9) {
        // Presque carré - grille 2x3
        const cellWidth = 0.47;
        const cellHeight = 0.3;
        const marginX = 0.025;
        const marginY = 0.05;
        
        for (let row = 0; row < 3; row++) {
          for (let col = 0; col < 2; col++) {
            panels.push({
              id: row * 2 + col,
              x: col * 0.5 + marginX,
              y: row * 0.33 + marginY,
              width: cellWidth,
              height: cellHeight,
              confidence: 0.7
            });
          }
        }
      } else {
        // Format portrait classique - grille verticale
        const panelWidth = 0.9;
        const panelHeight = 0.22;
        const marginX = 0.05;
        const marginY = 0.02;
        
        for (let i = 0; i < 4; i++) {
          panels.push({
            id: i,
            x: marginX,
            y: i * 0.245 + marginY,
            width: panelWidth,
            height: panelHeight,
            confidence: 0.7
          });
        }
      }
    }
    
    // S'assurer que tous les panels sont dans les limites [0,1]
    return panels.map(panel => ({
      ...panel,
      x: Math.max(0, Math.min(1 - panel.width, panel.x)),
      y: Math.max(0, Math.min(1 - panel.height, panel.y)),
      width: Math.max(0.1, Math.min(1, panel.width)),
      height: Math.max(0.1, Math.min(1, panel.height))
    }));
  }

  /**
   * Libère les ressources
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
      this.isInitialized = false;
      console.log('♻️ Détecteur libéré');
    }
  }
}