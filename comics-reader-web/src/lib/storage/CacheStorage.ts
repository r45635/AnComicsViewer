import { ComicsItem } from '@/types';

/**
 * Service de stockage utilisant le Cache API pour les gros fichiers
 * et localStorage pour les métadonnées légères
 */
export class CacheStorageService {
  private static readonly CACHE_NAME = 'comics-reader-cache';
  private static readonly METADATA_KEY = 'comics_metadata';

  /**
   * Sauvegarde un livre dans le cache
   */
  static async saveComicsItem(item: ComicsItem): Promise<void> {
    try {
      const cache = await caches.open(this.CACHE_NAME);
      
      // Stocker chaque page dans le cache
      for (let i = 0; i < item.pages.length; i++) {
        const pageUrl = item.pages[i];
        const cacheKey = `${item.id}/page-${i}`;
        
        // Créer une Response à partir de l'URL de la page
        const response = new Response(await this.dataUrlToBlob(pageUrl), {
          headers: { 'Content-Type': 'image/jpeg' }
        });
        
        await cache.put(cacheKey, response);
      }

      // Stocker les métadonnées légères dans localStorage
      await this.saveMetadata(item);
      
      console.log(`✅ Livre sauvegardé dans le cache: ${item.title}`);
    } catch (error) {
      console.error('❌ Erreur sauvegarde cache:', error);
      throw error;
    }
  }

  /**
   * Charge un livre depuis le cache
   */
  static async loadComicsItem(itemId: string): Promise<ComicsItem | null> {
    try {
      const metadata = await this.loadMetadata();
      const item = metadata.find(m => m.id === itemId);
      
      if (!item) return null;

      const cache = await caches.open(this.CACHE_NAME);
      const pages: string[] = [];

      // Récupérer toutes les pages depuis le cache
      for (let i = 0; i < item.pageCount; i++) {
        const cacheKey = `${itemId}/page-${i}`;
        const response = await cache.match(cacheKey);
        
        if (response) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          pages.push(url);
        }
      }

      return {
        ...item,
        pages
      };
    } catch (error) {
      console.error('❌ Erreur chargement cache:', error);
      return null;
    }
  }

  /**
   * Supprime un livre du cache
   */
  static async deleteComicsItem(itemId: string): Promise<void> {
    try {
      const cache = await caches.open(this.CACHE_NAME);
      const metadata = await this.loadMetadata();
      const item = metadata.find(m => m.id === itemId);
      
      if (item) {
        // Supprimer toutes les pages du cache
        for (let i = 0; i < item.pageCount; i++) {
          const cacheKey = `${itemId}/page-${i}`;
          await cache.delete(cacheKey);
        }

        // Mettre à jour les métadonnées
        const updatedMetadata = metadata.filter(m => m.id !== itemId);
        await this.saveAllMetadata(updatedMetadata);
      }

      console.log(`🗑️ Livre supprimé du cache: ${itemId}`);
    } catch (error) {
      console.error('❌ Erreur suppression cache:', error);
      throw error;
    }
  }

  /**
   * Charge toutes les métadonnées
   */
  static async loadMetadata(): Promise<Omit<ComicsItem, 'pages'>[]> {
    try {
      const stored = localStorage.getItem(this.METADATA_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('❌ Erreur chargement métadonnées:', error);
      return [];
    }
  }

  /**
   * Sauvegarde les métadonnées d'un livre
   */
  private static async saveMetadata(item: ComicsItem): Promise<void> {
    const metadata = await this.loadMetadata();
    const existingIndex = metadata.findIndex(m => m.id === item.id);
    
    // Créer l'objet métadonnées sans les pages
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    const { pages: _pages, ...metadataItem } = item;
    
    if (existingIndex >= 0) {
      metadata[existingIndex] = metadataItem;
    } else {
      metadata.push(metadataItem);
    }
    
    await this.saveAllMetadata(metadata);
  }

  /**
   * Sauvegarde toutes les métadonnées
   */
  private static async saveAllMetadata(metadata: Omit<ComicsItem, 'pages'>[]): Promise<void> {
    try {
      localStorage.setItem(this.METADATA_KEY, JSON.stringify(metadata));
    } catch (error) {
      console.error('❌ Erreur sauvegarde métadonnées:', error);
      throw error;
    }
  }

  /**
   * Convertit une Data URL en Blob
   */
  private static async dataUrlToBlob(dataUrl: string): Promise<Blob> {
    const response = await fetch(dataUrl);
    return response.blob();
  }

  /**
   * Calcule la taille utilisée par le cache
   */
  static async getCacheSize(): Promise<number> {
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        return estimate.usage || 0;
      }
      return 0;
    } catch (error) {
      console.error('❌ Erreur calcul taille cache:', error);
      return 0;
    }
  }

  /**
   * Nettoie le cache entier
   */
  static async clearCache(): Promise<void> {
    try {
      await caches.delete(this.CACHE_NAME);
      localStorage.removeItem(this.METADATA_KEY);
      console.log('🧹 Cache nettoyé');
    } catch (error) {
      console.error('❌ Erreur nettoyage cache:', error);
      throw error;
    }
  }
}