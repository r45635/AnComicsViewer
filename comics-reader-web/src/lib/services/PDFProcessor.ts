/**
 * Service de traitement et extraction de pages PDF pour le web
 * Utilise PDF.js et Canvas API pour le rendu
 */

import * as pdfjsLib from 'pdfjs-dist';
import { ComicsItem } from '@/types';

// Configuration PDF.js worker
if (typeof window !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.js';
}

interface ExtractionOptions {
  quality?: number;
  format?: 'png' | 'jpeg';
  startPage?: number;
  endPage?: number | null;
  maxPages?: number;
  maxWidth?: number;
  maxHeight?: number;
}

interface CacheInfo {
  pdfsInCache: number;
  totalFiles: number;
  totalSize: number;
}

// Interface pour les résultats d'extraction (future utilisation)
// interface PageExtractionResult {
//   pageNumber: number;
//   dataUrl: string;
//   width: number;
//   height: number;
//   size: number;
// }

export class PDFProcessor {
  private extractedPages = new Map<string, string[]>();
  private cacheSizeLimit = 100 * 1024 * 1024; // 100MB limit

  constructor() {
    this.initialize();
  }

  /**
   * Initialise le processeur PDF
   */
  async initialize(): Promise<void> {
    try {
      console.log('✅ PDFProcessor web initialisé');
    } catch (error) {
      console.error('❌ Erreur initialisation PDFProcessor:', error);
      throw error;
    }
  }

  /**
   * Extrait toutes les pages d'un PDF en images
   */
  async extractPagesFromPDF(
    pdfFile: File | Blob | ArrayBuffer,
    options: ExtractionOptions = {}
  ): Promise<string[]> {
    const {
      quality = 2.5, // Scale factor plus élevé pour une meilleure qualité
      format = 'png',
      startPage = 1,
      endPage = null,
      maxPages = 100,
      maxWidth = 2560, // Résolution plus haute pour préserver les détails
      maxHeight = 2560,
    } = options;

    try {
      console.log('📄 Extraction PDF en cours...');
      
      // Générer un ID unique pour ce PDF
      const pdfId = await this.generatePDFId(pdfFile);
      
      // Vérifier si déjà extrait
      if (this.extractedPages.has(pdfId)) {
        console.log('📋 Pages déjà extraites, utilisation du cache');
        return this.extractedPages.get(pdfId)!;
      }

      // Charger le PDF avec PDF.js
      const arrayBuffer = await this.fileToArrayBuffer(pdfFile);
      const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
      const pdf = await loadingTask.promise;

      console.log(`📊 PDF chargé: ${pdf.numPages} pages`);

      // Calculer les pages à extraire
      const actualEndPage = endPage ? Math.min(endPage, pdf.numPages) : pdf.numPages;
      const pagesToExtract = Math.min(actualEndPage - startPage + 1, maxPages);
      
      const extractedUris: string[] = [];
      
      // Extraire chaque page
      for (let pageNum = startPage; pageNum <= startPage + pagesToExtract - 1; pageNum++) {
        try {
          const page = await pdf.getPage(pageNum);
          const dataUrl = await this.renderPDFPageToCanvas(page, {
            quality,
            format,
            maxWidth,
            maxHeight
          });
          
          extractedUris.push(dataUrl);
          console.log(`✅ Page ${pageNum} extraite`);
          
        } catch (pageError) {
          console.error(`❌ Erreur extraction page ${pageNum}:`, pageError);
          // Continuer avec les autres pages
        }
      }

      // Mettre en cache
      this.extractedPages.set(pdfId, extractedUris);
      await this.enforceStorageLimit();
      
      console.log(`✅ ${extractedUris.length} pages extraites du PDF`);
      return extractedUris;

    } catch (error) {
      console.error('❌ Erreur extraction PDF:', error);
      throw error;
    }
  }

  /**
   * Rendu d'une page PDF sur canvas
   */
  private async renderPDFPageToCanvas(
    page: pdfjsLib.PDFPageProxy,
    options: {
      quality: number;
      format: string;
      maxWidth: number;
      maxHeight: number;
    }
  ): Promise<string> {
    const viewport = page.getViewport({ scale: 1.0 });
    
    // Calculer l'échelle pour respecter les limites de taille
    const scaleX = options.maxWidth / viewport.width;
    const scaleY = options.maxHeight / viewport.height;
    const scale = Math.min(scaleX, scaleY, options.quality);
    
    const scaledViewport = page.getViewport({ scale });
    
    // Créer le canvas
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    if (!context) {
      throw new Error('Impossible de créer le contexte canvas');
    }
    
    canvas.width = scaledViewport.width;
    canvas.height = scaledViewport.height;
    
    // Configuration du rendu
    const renderContext = {
      canvasContext: context,
      viewport: scaledViewport,
      background: 'white', // Fond blanc pour les PDFs
      canvas: canvas,
    };
    
    // Rendu de la page
    await page.render(renderContext).promise;
    
    // Convertir en Data URL avec qualité préservée
    const mimeType = 'image/jpeg'; // JPEG pour équilibrer qualité/taille
    const quality = 0.85; // Qualité élevée pour préserver les détails BD
    
    return canvas.toDataURL(mimeType, quality);
  }

  /**
   * Convertit un File/Blob en ArrayBuffer
   */
  private fileToArrayBuffer(file: File | Blob | ArrayBuffer): Promise<ArrayBuffer> {
    if (file instanceof ArrayBuffer) {
      return Promise.resolve(file);
    }
    
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as ArrayBuffer);
      reader.onerror = reject;
      reader.readAsArrayBuffer(file);
    });
  }

  /**
   * Génère un ID unique pour un PDF
   */
  private async generatePDFId(file: File | Blob | ArrayBuffer): Promise<string> {
    let data: ArrayBuffer;
    
    if (file instanceof ArrayBuffer) {
      data = file;
    } else {
      data = await this.fileToArrayBuffer(file);
    }
    
    // Utiliser les premiers bytes pour générer un hash simple
    const view = new Uint8Array(data, 0, Math.min(1024, data.byteLength));
    let hash = 0;
    
    for (let i = 0; i < view.length; i++) {
      hash = ((hash << 5) - hash) + view[i];
      hash = hash & hash; // Convert to 32-bit integer
    }
    
    return `pdf_${Math.abs(hash)}_${data.byteLength}`;
  }

  /**
   * Crée un ComicsItem à partir d'un PDF
   */
  async createComicsItemFromPDF(
    file: File,
    options: ExtractionOptions = {}
  ): Promise<ComicsItem> {
    const pages = await this.extractPagesFromPDF(file, options);
    
    const item: ComicsItem = {
      id: `pdf_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: file.name.replace('.pdf', ''),
      type: 'pdf',
      originalUri: URL.createObjectURL(file),
      pages,
      pageCount: pages.length,
      addedDate: new Date().toISOString(),
      thumbnail: pages[0] || undefined,
    };
    
    return item;
  }

  /**
   * Crée un ComicsItem à partir d'une image
   */
  async createComicsItemFromImage(file: File): Promise<ComicsItem> {
    // Optimiser l'image pour économiser l'espace
    const optimizedImageUrl = await this.optimizeImageForStorage(file);
    
    const item: ComicsItem = {
      id: `img_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      title: file.name.replace(/\.(jpg|jpeg|png|gif|webp)$/i, ''),
      type: 'image',
      originalUri: optimizedImageUrl,
      pages: [optimizedImageUrl],
      pageCount: 1,
      addedDate: new Date().toISOString(),
      thumbnail: optimizedImageUrl,
    };
    
    return item;
  }

  /**
   * Optimise une image pour le stockage avec compression agressive
   */
  async optimizeImageForStorage(file: File): Promise<string> {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        try {
          // Calculer les nouvelles dimensions avec une limite plus généreuse pour préserver la qualité
          const maxWidth = 1200;
          const maxHeight = 1600;
          const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1);
          const newWidth = Math.floor(img.width * scale);
          const newHeight = Math.floor(img.height * scale);

          // Créer un canvas pour le redimensionnement
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');

          if (!ctx) {
            resolve(URL.createObjectURL(file));
            return;
          }

          canvas.width = newWidth;
          canvas.height = newHeight;

          // Redimensionner l'image
          ctx.drawImage(img, 0, 0, newWidth, newHeight);

          // Convertir en Data URL avec qualité équilibrée
          const optimizedDataUrl = canvas.toDataURL('image/jpeg', 0.8);
          
          console.log(`🔧 Image optimisée pour stockage: ${img.width}x${img.height} → ${newWidth}x${newHeight}`);
          resolve(optimizedDataUrl);

        } catch (error) {
          console.error('❌ Erreur optimisation image:', error);
          resolve(URL.createObjectURL(file)); // Fallback
        }
      };

      img.onerror = () => {
        console.error('❌ Erreur chargement image pour optimisation');
        resolve(URL.createObjectURL(file)); // Fallback
      };

      img.src = URL.createObjectURL(file);
    });
  }

  /**
   * Optimise une image pour la détection
   */
  async optimizeImageForDetection(
    imageDataUrl: string,
    options: {
      maxWidth?: number;
      maxHeight?: number;
      quality?: number;
    } = {}
  ): Promise<string> {
    const {
      maxWidth = 1280,
      maxHeight = 1280,
      quality = 0.8,
    } = options;

    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        try {
          // Si l'image est déjà petite, la retourner telle quelle
          if (img.width <= maxWidth && img.height <= maxHeight) {
            resolve(imageDataUrl);
            return;
          }

          // Calculer les nouvelles dimensions
          const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
          const newWidth = Math.floor(img.width * scale);
          const newHeight = Math.floor(img.height * scale);

          // Créer un canvas pour le redimensionnement
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');

          if (!ctx) {
            resolve(imageDataUrl);
            return;
          }

          canvas.width = newWidth;
          canvas.height = newHeight;

          // Redimensionner l'image
          ctx.drawImage(img, 0, 0, newWidth, newHeight);

          // Convertir en Data URL optimisée
          const optimizedDataUrl = canvas.toDataURL('image/jpeg', quality);
          
          console.log(`🔧 Image optimisée: ${img.width}x${img.height} → ${newWidth}x${newHeight}`);
          resolve(optimizedDataUrl);

        } catch (error) {
          console.error('❌ Erreur optimisation image:', error);
          resolve(imageDataUrl); // Fallback
        }
      };

      img.onerror = () => {
        console.error('❌ Erreur chargement image pour optimisation');
        resolve(imageDataUrl); // Fallback
      };

      img.src = imageDataUrl;
    });
  }

  /**
   * Applique une limite de stockage en mémoire
   */
  private async enforceStorageLimit(): Promise<void> {
    // Calculer la taille approximative du cache
    let totalSize = 0;
    const entries = Array.from(this.extractedPages.entries());
    
    for (const [, pages] of entries) {
      for (const page of pages) {
        // Estimation approximative de la taille d'un data URL
        totalSize += page.length * 0.75; // Base64 encoding overhead
      }
    }

    // Si on dépasse la limite, supprimer les plus anciens
    if (totalSize > this.cacheSizeLimit) {
      console.log('🧹 Limite de cache atteinte, nettoyage...');
      
      // Supprimer la première moitié des entrées (FIFO)
      const toRemove = Math.ceil(entries.length / 2);
      for (let i = 0; i < toRemove; i++) {
        this.extractedPages.delete(entries[i][0]);
      }
    }
  }

  /**
   * Nettoie le cache des pages extraites
   */
  async clearCache(pdfId?: string): Promise<void> {
    try {
      if (pdfId) {
        this.extractedPages.delete(pdfId);
      } else {
        this.extractedPages.clear();
      }
      
      console.log('🧹 Cache PDF nettoyé');
      
    } catch (error) {
      console.error('❌ Erreur nettoyage cache:', error);
    }
  }

  /**
   * Obtient les informations sur l'utilisation du cache
   */
  async getCacheInfo(): Promise<CacheInfo> {
    try {
      let totalFiles = 0;
      let totalSize = 0;

      for (const [, pages] of this.extractedPages.entries()) {
        totalFiles += pages.length;
        for (const page of pages) {
          totalSize += page.length * 0.75; // Estimation taille
        }
      }

      return {
        pdfsInCache: this.extractedPages.size,
        totalFiles,
        totalSize,
      };
      
    } catch (error) {
      console.error('❌ Erreur info cache:', error);
      return { pdfsInCache: 0, totalFiles: 0, totalSize: 0 };
    }
  }

  /**
   * Valide si un fichier est un PDF
   */
  static isPDFFile(file: File): boolean {
    return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
  }

  /**
   * Valide si un fichier est une image supportée
   */
  static isImageFile(file: File): boolean {
    const supportedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp'];
    return supportedTypes.includes(file.type) || 
           /\.(jpg|jpeg|png|gif|webp)$/i.test(file.name);
  }
}