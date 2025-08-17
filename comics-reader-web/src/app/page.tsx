'use client';

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Upload, BookOpen, FileImage, Trash2, Calendar, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ComicsItem } from '@/types';
import { LoadingIndicator } from '@/components/comics/LoadingIndicator';
import { CacheStorageService } from '@/lib/storage/CacheStorage';
import { toast } from 'sonner';

// Pas besoin d'import global, on fait du dynamic import dans les fonctions


export default function HomePage() {
  const router = useRouter();
  const [library, setLibrary] = useState<ComicsItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingText, setLoadingText] = useState('Initialisation...');
  const [pdfProcessor, setPdfProcessor] = useState<any>(null); // eslint-disable-line @typescript-eslint/no-explicit-any
  const [storageInfo, setStorageInfo] = useState<{used: number, total: number} | null>(null);

  useEffect(() => {
    initializeApp();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Initialise l'application
   */
  const initializeApp = async () => {
    try {
      setIsLoading(true);
      setLoadingText('Initialisation...');

      // Initialiser le processeur PDF (côté client seulement)
      if (typeof window !== 'undefined') {
        const { PDFProcessor: PDFProcessorClass } = await import('@/lib/services/PDFProcessor');
        const processor = new PDFProcessorClass();
        setPdfProcessor(processor);
      }

      // Charger la bibliothèque depuis le cache
      await loadLibrary();

      // Vérifier l'espace de stockage
      await checkStorageQuota();

    } catch (error) {
      console.error('❌ Erreur initialisation:', error);
      toast.error('Impossible d\'initialiser l\'application');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Charge la bibliothèque depuis le cache
   */
  const loadLibrary = async () => {
    try {
      const metadata = await CacheStorageService.loadMetadata();
      setLibrary(metadata as ComicsItem[]);
      console.log(`📚 Bibliothèque chargée: ${metadata.length} éléments`);
    } catch (error) {
      console.error('❌ Erreur chargement bibliothèque:', error);
      toast.error('Erreur lors du chargement de la bibliothèque');
    }
  };

  /**
   * Sauvegarde un livre dans le cache
   */
  const saveLibrary = async (newLibrary: ComicsItem[]) => {
    try {
      // Sauvegarder seulement le nouveau livre (le dernier ajouté)
      const newItem = newLibrary[newLibrary.length - 1];
      if (newItem && newItem.pages && newItem.pages.length > 0) {
        await CacheStorageService.saveComicsItem(newItem);
        console.log(`💾 Livre sauvegardé: ${newItem.title}`);
        toast.success('Livre ajouté à la bibliothèque');
      }
      
      // Mettre à jour la liste locale
      setLibrary(newLibrary);
    } catch (error) {
      console.error('❌ Erreur sauvegarde bibliothèque:', error);
      toast.error('Erreur lors de la sauvegarde');
      throw error;
    }
  };

  /**
   * Vérifie l'espace de stockage disponible
   */
  const checkStorageQuota = async () => {
    try {
      const used = await CacheStorageService.getCacheSize();
      
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimate = await navigator.storage.estimate();
        const total = estimate.quota || 0;
        setStorageInfo({ used, total });
        
        // Alerter si proche de la limite
        if (used > total * 0.8) {
          toast.warning('Espace de stockage bientôt plein. Supprimez des éléments anciens.');
        }
      }
    } catch (error) {
      console.warn('⚠️ Impossible de vérifier l\'espace de stockage:', error);
    }
  };

  /**
   * Formate la taille en format lisible
   */
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  /**
   * Gestionnaire de drag & drop
   */
  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = async (event: React.DragEvent) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      await processFile(files[0]);
    }
  };

  /**
   * Gestionnaire de clic sur le bouton
   */
  const handleButtonClick = () => {
    console.log('🖱️ Bouton cliqué');
    try {
      const input = document.getElementById('file-input') as HTMLInputElement;
      console.log('📄 Input trouvé:', !!input);
      if (input) {
        console.log('🔄 Déclenchement du click sur input');
        input.click();
      } else {
        console.error('❌ Élément input non trouvé');
        toast.error('Erreur d\'interface - rechargez la page');
      }
    } catch (error) {
      console.error('❌ Erreur lors du clic:', error);
      toast.error('Impossible d\'ouvrir le sélecteur de fichiers');
    }
  };

  /**
   * Gestionnaire de sélection de fichiers
   */
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    console.log('📁 Fichier sélectionné via input');
    const files = event.target.files;
    if (!files || files.length === 0) {
      console.log('❌ Aucun fichier sélectionné');
      return;
    }

    console.log('✅ Traitement du fichier:', files[0].name);
    await processFile(files[0]);

    // Reset input
    event.target.value = '';
  };

  /**
   * Traite un fichier (commun pour drag&drop et sélection)
   */
  const processFile = async (file: File) => {
    console.log('📄 Fichier sélectionné:', file.name);

    try {
      if (!pdfProcessor) {
        toast.error('Processeur PDF non disponible');
        return;
      }
      
      // Importer dynamiquement pour les méthodes statiques
      const { PDFProcessor: PDFProcessorClass } = await import('@/lib/services/PDFProcessor');
      
      if (PDFProcessorClass.isPDFFile(file)) {
        await processPDF(file);
      } else if (PDFProcessorClass.isImageFile(file)) {
        await processImage(file);
      } else {
        toast.error('Format de fichier non supporté');
      }
    } catch (error) {
      console.error('❌ Erreur traitement fichier:', error);
      toast.error('Impossible de traiter le fichier');
    }
  };

  /**
   * Traite un fichier PDF
   */
  const processPDF = async (file: File) => {
    if (!pdfProcessor) return;
    
    try {
      setIsLoading(true);
      setLoadingText('Extraction des pages PDF...');

      const libraryItem = await pdfProcessor.createComicsItemFromPDF(file, {
        quality: 1.0, // Réduire encore plus la qualité pour économiser l'espace
        maxPages: 10,  // Limiter davantage le nombre de pages
        maxWidth: 800, // Réduire encore plus la résolution
        maxHeight: 800,
      });

      // Ajouter à la bibliothèque
      const newLibrary = [...library, libraryItem];
      setLibrary(newLibrary);
      await saveLibrary(newLibrary);

      console.log(`✅ PDF ajouté: ${libraryItem.pageCount} pages`);
      toast.success(`PDF ajouté avec ${libraryItem.pageCount} pages`);
      
      // Naviguer vers le lecteur
      openLibraryItem(libraryItem);

    } catch (error) {
      console.error('❌ Erreur traitement PDF:', error);
      toast.error('Impossible de traiter le PDF');
    } finally {
      setIsLoading(false);
    }
  };

  /**
   * Traite une image
   */
  const processImage = async (file: File) => {
    if (!pdfProcessor) return;
    
    try {
      const libraryItem = await pdfProcessor.createComicsItemFromImage(file);

      // Ajouter à la bibliothèque
      const newLibrary = [...library, libraryItem];
      setLibrary(newLibrary);
      await saveLibrary(newLibrary);

      console.log('✅ Image ajoutée');
      toast.success('Image ajoutée à la bibliothèque');
      
      // Naviguer vers le lecteur
      openLibraryItem(libraryItem);

    } catch (error) {
      console.error('❌ Erreur traitement image:', error);
      toast.error('Impossible de traiter l\'image');
    }
  };

  /**
   * Ouvre un élément de la bibliothèque
   */
  const openLibraryItem = (item: ComicsItem) => {
    // Passer seulement l'ID maintenant que les données sont dans le cache
    router.push(`/reader?id=${item.id}`);
  };

  /**
   * Supprime un élément de la bibliothèque
   */
  const deleteLibraryItem = async (itemId: string) => {
    try {
      // Supprimer du cache
      await CacheStorageService.deleteComicsItem(itemId);
      
      // Mettre à jour la liste locale
      const newLibrary = library.filter(item => item.id !== itemId);
      setLibrary(newLibrary);
      
      console.log(`🗑️ Élément supprimé: ${itemId}`);
      toast.success('Élément supprimé de la bibliothèque');
      
      // Mettre à jour les informations de stockage
      await checkStorageQuota();
    } catch (error) {
      console.error('❌ Erreur suppression:', error);
      toast.error('Erreur lors de la suppression');
    }
  };

  /**
   * Formate la date d'ajout
   */
  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('fr-FR', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header avec gradient */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white">
        <div className="container mx-auto px-6 py-8">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <h1 className="text-3xl font-bold mb-2">Comics Reader Web</h1>
              <p className="text-blue-100">
                Détection IA • Multi-BD • {library.length} élément{library.length > 1 ? 's' : ''}
                {storageInfo && (
                  <span className="ml-2 opacity-75">
                    • {formatBytes(storageInfo.used)} / {formatBytes(storageInfo.total)}
                  </span>
                )}
              </p>
            </div>
            <div className="bg-white/20 rounded-full p-4">
              <span className="text-3xl">🤖</span>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Bouton d'ajout principal */}
        <Card 
          className="mb-8 border-2 border-dashed border-gray-300 hover:border-blue-400 transition-colors"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          <CardContent className="p-8 text-center">
            <input
              type="file"
              accept=".pdf,.jpg,.jpeg,.png,.gif,.webp"
              onChange={handleFileSelect}
              className="hidden"
              id="file-input"
            />
            <Button 
              size="lg" 
              onClick={handleButtonClick}
              className="bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600"
            >
              <Upload className="mr-2 h-5 w-5" />
              Ajouter PDF ou Image
            </Button>
            <p className="text-sm text-gray-600 mt-4">
              Glissez-déposez vos fichiers ou cliquez pour sélectionner
            </p>
          </CardContent>
        </Card>

        {/* Liste de la bibliothèque */}
        {library.length > 0 ? (
          <div>
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-6">
              Ma Bibliothèque
            </h2>
            
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {library.map((item) => (
                <Card 
                  key={item.id} 
                  className="group hover:shadow-lg transition-all duration-200 cursor-pointer"
                  onClick={() => openLibraryItem(item)}
                >
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <CardTitle className="text-lg truncate" title={item.title}>
                          {item.title}
                        </CardTitle>
                        <div className="flex items-center gap-2 mt-2">
                          <Badge variant="secondary" className="flex items-center gap-1">
                            {item.type === 'pdf' ? (
                              <BookOpen className="h-3 w-3" />
                            ) : (
                              <FileImage className="h-3 w-3" />
                            )}
                            {item.type.toUpperCase()}
                          </Badge>
                          <span className="text-sm text-gray-600">
                            {item.pageCount} page{item.pageCount > 1 ? 's' : ''}
                          </span>
                        </div>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        className="opacity-0 group-hover:opacity-100 transition-opacity"
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteLibraryItem(item.id);
                        }}
                      >
                        <Trash2 className="h-4 w-4 text-red-500" />
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className="pt-0">
                    {/* Thumbnail */}
                    {item.thumbnail && (
                      <div className="relative h-40 bg-gray-100 rounded-md overflow-hidden mb-3">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={item.thumbnail}
                          alt={item.title}
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                          <Eye className="h-6 w-6 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                        </div>
                      </div>
                    )}

                    <Separator className="mb-3" />

                    {/* Informations */}
                    <div className="flex items-center text-sm text-gray-600">
                      <Calendar className="h-4 w-4 mr-1" />
                      {formatDate(item.addedDate)}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ) : (
          /* État vide */
          <Card className="border-none shadow-lg">
            <CardContent className="p-12 text-center">
              <div className="text-6xl mb-6">📚</div>
              <h3 className="text-2xl font-bold text-gray-800 dark:text-white mb-3">
                Bibliothèque vide
              </h3>
              <p className="text-gray-600 dark:text-gray-400 max-w-md mx-auto leading-relaxed">
                Ajoutez votre première BD ou manga.{' '}
                La détection IA analysera automatiquement{' '}
                les panels pour une lecture optimale.
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Indicateur de chargement */}
      <LoadingIndicator
        visible={isLoading}
        isLoading={isLoading}
        text={loadingText}
        subText="Intelligence artificielle en action..."
      />
    </div>
  );
}
