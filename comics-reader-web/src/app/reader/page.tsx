'use client';

import React, { useState, useEffect, useCallback, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { ArrowLeft, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Eye, EyeOff, Settings, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { ComicsViewer } from '@/components/comics/ComicsViewer';
import { LoadingIndicator } from '@/components/comics/LoadingIndicator';
import { CacheStorageService } from '@/lib/storage/CacheStorage';
import { ComicsItem, Panel } from '@/types';
import { toast } from 'sonner';

function ReaderPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [comicsItem, setComicsItem] = useState<ComicsItem | null>(null);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [panels, setPanels] = useState<Panel[]>([]);
  const [isDetecting] = useState(false);
  const [showPanels, setShowPanels] = useState(true);
  const [autoDetect, setAutoDetect] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Paramètres de lecture
  const [readingSettings, setReadingSettings] = useState({
    autoAdvance: false,
    autoAdvanceDelay: 3,
    fitToScreen: true,
    darkMode: false,
  });

  useEffect(() => {
    initializeReader();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          goToPreviousPage();
          break;
        case 'ArrowRight':
          event.preventDefault();
          goToNextPage();
          break;
        case 'Escape':
          if (isFullscreen) {
            exitFullscreen();
          }
          break;
        case 'f':
        case 'F':
          toggleFullscreen();
          break;
        case 'p':
        case 'P':
          setShowPanels(!showPanels);
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [currentPageIndex, comicsItem, isFullscreen, showPanels]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Initialise le lecteur avec les données de l'URL
   */
  const initializeReader = async () => {
    try {
      // Nouveau: récupérer l'ID depuis l'URL
      const itemId = searchParams.get('id');
      if (!itemId) {
        // Fallback: support de l'ancien format pour compatibilité
        const itemParam = searchParams.get('item');
        if (itemParam) {
          const decodedItem = JSON.parse(atob(itemParam));
          const fullItem = await CacheStorageService.loadComicsItem(decodedItem.id);
          if (fullItem) {
            setComicsItem(fullItem);
            console.log('📖 Lecteur initialisé (mode legacy):', fullItem.title);
            return;
          }
        }
        
        toast.error('Aucun élément à lire');
        router.push('/');
        return;
      }

      // Charger directement depuis le cache avec l'ID
      const fullItem = await CacheStorageService.loadComicsItem(itemId);
      if (fullItem) {
        setComicsItem(fullItem);
        console.log('📖 Lecteur initialisé:', fullItem.title);
      } else {
        toast.error('Livre non trouvé dans le cache');
        router.push('/');
      }
      
    } catch (error) {
      console.error('❌ Erreur initialisation lecteur:', error);
      toast.error('Impossible de charger l&apos;élément');
      router.push('/');
    }
  };

  /**
   * Gestionnaire de détection de panels
   */
  const handlePanelsDetected = useCallback((detectedPanels: Panel[]) => {
    setPanels(detectedPanels);
    console.log(`✅ ${detectedPanels.length} panels détectés pour la page ${currentPageIndex + 1}`);
  }, [currentPageIndex]);

  /**
   * Gestionnaire de sélection de panel
   */
  const handlePanelSelect = useCallback((panel: Panel, index: number) => {
    console.log(`👆 Panel ${index + 1} sélectionné`);
    toast.success(`Panel ${index + 1} sélectionné`);
  }, []);

  /**
   * Navigation vers la page suivante
   */
  const goToNextPage = useCallback(() => {
    if (!comicsItem) return;
    
    if (currentPageIndex < comicsItem.pages.length - 1) {
      setCurrentPageIndex(currentPageIndex + 1);
      setPanels([]); // Reset panels pour la nouvelle page
    } else {
      toast.info('Dernière page atteinte');
    }
  }, [currentPageIndex, comicsItem]);

  /**
   * Navigation vers la page précédente
   */
  const goToPreviousPage = useCallback(() => {
    if (currentPageIndex > 0) {
      setCurrentPageIndex(currentPageIndex - 1);
      setPanels([]); // Reset panels pour la nouvelle page
    } else {
      toast.info('Première page atteinte');
    }
  }, [currentPageIndex]);

  /**
   * Aller à une page spécifique
   */
  const goToPage = useCallback((pageIndex: number) => {
    if (!comicsItem) return;
    
    const clampedIndex = Math.max(0, Math.min(pageIndex, comicsItem.pages.length - 1));
    setCurrentPageIndex(clampedIndex);
    setPanels([]);
  }, [comicsItem]);

  /**
   * Toggle mode plein écran
   */
  const toggleFullscreen = async () => {
    try {
      if (!isFullscreen) {
        await document.documentElement.requestFullscreen();
        setIsFullscreen(true);
      } else {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      console.error('❌ Erreur fullscreen:', error);
    }
  };

  /**
   * Sortir du mode plein écran
   */
  const exitFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        setIsFullscreen(false);
      }
    } catch (error) {
      console.error('❌ Erreur sortie fullscreen:', error);
    }
  };

  /**
   * Retour à l'accueil
   */
  const goHome = () => {
    router.push('/');
  };

  if (!comicsItem) {
    return (
      <LoadingIndicator
        visible={true}
        isLoading={true}
        text="Chargement du lecteur..."
      />
    );
  }

  const currentPageUri = comicsItem.pages[currentPageIndex];

  return (
    <div className={`min-h-screen ${isFullscreen ? 'bg-black' : 'bg-gray-900'}`}>
      {/* Barre de contrôle supérieure */}
      {!isFullscreen && (
        <div className="bg-white dark:bg-gray-800 border-b shadow-sm">
          <div className="container mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              {/* Navigation et titre */}
              <div className="flex items-center gap-4">
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={goHome}
                  className="flex items-center gap-2"
                >
                  <Home className="h-4 w-4" />
                  <ArrowLeft className="h-4 w-4" />
                  Accueil
                </Button>
                
                <div className="flex-1">
                  <h1 className="font-semibold text-lg truncate">{comicsItem.title}</h1>
                  <p className="text-sm text-gray-600">
                    Page {currentPageIndex + 1} sur {comicsItem.pages.length}
                    {panels.length > 0 && ` • ${panels.length} panels détectés`}
                  </p>
                </div>
              </div>

              {/* Contrôles */}
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="flex items-center gap-1">
                  {comicsItem.type.toUpperCase()}
                </Badge>

                {/* Paramètres */}
                <Sheet>
                  <SheetTrigger asChild>
                    <Button variant="ghost" size="sm">
                      <Settings className="h-4 w-4" />
                    </Button>
                  </SheetTrigger>
                  <SheetContent>
                    <SheetHeader>
                      <SheetTitle>Paramètres de lecture</SheetTitle>
                    </SheetHeader>
                    <div className="space-y-6 mt-6">
                      {/* Affichage des panels */}
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium">Afficher les panels</label>
                          <p className="text-xs text-gray-600">
                            Activer la détection et l&apos;affichage des panels
                          </p>
                        </div>
                        <Switch
                          checked={showPanels}
                          onCheckedChange={setShowPanels}
                        />
                      </div>

                      {/* Détection automatique */}
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium">Détection automatique</label>
                          <p className="text-xs text-gray-600">
                            Détecter automatiquement les panels
                          </p>
                        </div>
                        <Switch
                          checked={autoDetect}
                          onCheckedChange={setAutoDetect}
                        />
                      </div>

                      {/* Mode sombre */}
                      <div className="flex items-center justify-between">
                        <div>
                          <label className="text-sm font-medium">Mode sombre</label>
                          <p className="text-xs text-gray-600">
                            Interface sombre pour la lecture
                          </p>
                        </div>
                        <Switch
                          checked={readingSettings.darkMode}
                          onCheckedChange={(checked) =>
                            setReadingSettings(prev => ({ ...prev, darkMode: checked }))
                          }
                        />
                      </div>

                      {/* Navigation rapide */}
                      <div className="space-y-3">
                        <label className="text-sm font-medium">Navigation rapide</label>
                        <div className="space-y-2">
                          <p className="text-xs text-gray-600">Page {currentPageIndex + 1}</p>
                          <Slider
                            value={[currentPageIndex]}
                            onValueChange={([value]) => goToPage(value)}
                            max={comicsItem.pages.length - 1}
                            step={1}
                            className="w-full"
                          />
                        </div>
                      </div>
                    </div>
                  </SheetContent>
                </Sheet>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Zone de lecture principale */}
      <div className="relative flex-1">
        <ComicsViewer
          imageUri={currentPageUri}
          showPanels={showPanels}
          autoDetect={autoDetect}
          panels={panels}
          onPanelSelect={handlePanelSelect}
          onPanelsDetected={handlePanelsDetected}
          className="w-full h-[calc(100vh-80px)]"
        />

        {/* Contrôles de navigation flottants */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Navigation gauche */}
          <div className="absolute left-4 top-1/2 transform -translate-y-1/2">
            <Button
              variant="secondary"
              size="lg"
              className="pointer-events-auto opacity-70 hover:opacity-100 transition-opacity"
              onClick={goToPreviousPage}
              disabled={currentPageIndex === 0}
            >
              <ChevronLeft className="h-6 w-6" />
            </Button>
          </div>

          {/* Navigation droite */}
          <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
            <Button
              variant="secondary"
              size="lg"
              className="pointer-events-auto opacity-70 hover:opacity-100 transition-opacity"
              onClick={goToNextPage}
              disabled={currentPageIndex === comicsItem.pages.length - 1}
            >
              <ChevronRight className="h-6 w-6" />
            </Button>
          </div>

          {/* Contrôles en bas */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2">
            <Card className="pointer-events-auto">
              <CardContent className="flex items-center gap-2 p-3">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowPanels(!showPanels)}
                  title={showPanels ? 'Masquer les panels' : 'Afficher les panels'}
                >
                  {showPanels ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleFullscreen}
                  title="Mode plein écran (F)"
                >
                  {isFullscreen ? <ZoomOut className="h-4 w-4" /> : <ZoomIn className="h-4 w-4" />}
                </Button>

                <div className="text-sm font-medium px-2">
                  {currentPageIndex + 1} / {comicsItem.pages.length}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Zones de navigation invisible (tap zones) */}
        <div
          className="absolute left-0 top-0 w-1/3 h-full cursor-pointer"
          onClick={goToPreviousPage}
          title="Page précédente"
        />
        <div
          className="absolute right-0 top-0 w-1/3 h-full cursor-pointer"
          onClick={goToNextPage}
          title="Page suivante"
        />
      </div>

      {/* Indicateur de chargement */}
      <LoadingIndicator
        visible={isDetecting}
        isLoading={isDetecting}
        text="Détection IA en cours..."
        subText="Analyse des panels de la page..."
      />
    </div>
  );
}

export default function ReaderPage() {
  return (
    <Suspense fallback={
      <LoadingIndicator
        visible={true}
        isLoading={true}
        text="Chargement du lecteur..."
        subText="Préparation de l&apos;interface..."
      />
    }>
      <ReaderPageContent />
    </Suspense>
  );
}