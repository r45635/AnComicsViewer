'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, useMotionValue, useTransform, PanInfo } from 'framer-motion';
import Image from 'next/image';
import { Panel, ImageSize } from '@/types';
import { PanelOverlay } from './PanelOverlay';
import { LoadingIndicator } from './LoadingIndicator';

interface ComicsViewerProps {
  imageUri: string;
  onPanelSelect?: (panel: Panel, index: number) => void;
  showPanels?: boolean;
  autoDetect?: boolean;
  panels?: Panel[];
  onPanelsDetected?: (panels: Panel[]) => void;
  className?: string;
}

export const ComicsViewer: React.FC<ComicsViewerProps> = ({
  imageUri,
  onPanelSelect,
  showPanels = true,
  autoDetect = true,
  panels: externalPanels,
  onPanelsDetected,
  className = ''
}) => {
  // État du composant
  const [panels, setPanels] = useState<Panel[]>(externalPanels || []);
  const [selectedPanel, setSelectedPanel] = useState<Panel | null>(null);
  const [currentPanelIndex, setCurrentPanelIndex] = useState(-1);
  const [isDetecting, setIsDetecting] = useState(false);
  const [imageSize, setImageSize] = useState<ImageSize>({ width: 0, height: 0 });
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  // Références
  const containerRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  // Valeurs animées pour zoom/pan
  const scale = useMotionValue(1);
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  // Transformations
  const transform = useTransform(
    [scale, x, y],
    ([s, xPos, yPos]) => `scale(${s}) translate(${xPos}px, ${yPos}px)`
  );

  // Détection des panels quand l'image change
  useEffect(() => {
    if (externalPanels) {
      setPanels(externalPanels);
    } else if (imageUri && isImageLoaded && autoDetect) {
      detectPanels();
    }
  }, [imageUri, isImageLoaded, autoDetect, externalPanels]); // eslint-disable-line react-hooks/exhaustive-deps

  // Mise à jour de la taille du conteneur
  useEffect(() => {
    const updateContainerSize = () => {
      if (containerRef.current) {
        const { clientWidth, clientHeight } = containerRef.current;
        setContainerSize({ width: clientWidth, height: clientHeight });
      }
    };

    updateContainerSize();
    window.addEventListener('resize', updateContainerSize);
    return () => window.removeEventListener('resize', updateContainerSize);
  }, []);

  /**
   * Détection des panels avec IA
   */
  const detectPanels = async () => {
    setIsDetecting(true);
    setError(null);

    try {
      console.log('🔍 Début détection panels...', { imageUri, imageSize, autoDetect, showPanels });
      
      // Import dynamique du détecteur pour éviter les erreurs SSR
      const { EmbeddedPanelDetector } = await import('@/lib/ml/PanelDetector');
      const detector = new EmbeddedPanelDetector();
      
      const detectedPanels = await detector.detectPanels(imageUri);
      console.log('🔍 Panels bruts détectés:', detectedPanels);
      
      // Convertir les coordonnées relatives en absolues
      const scaledPanels = detectedPanels.map((panel, index) => ({
        ...panel,
        id: index,
        x: panel.x * imageSize.width,
        y: panel.y * imageSize.height,
        width: panel.width * imageSize.width,
        height: panel.height * imageSize.height,
      }));

      console.log('📐 Panels mis à l\'échelle:', scaledPanels);
      setPanels(scaledPanels);
      setCurrentPanelIndex(-1);
      console.log(`✅ ${scaledPanels.length} panels définis dans le state`);

      if (onPanelsDetected) {
        onPanelsDetected(scaledPanels);
      }

    } catch (error) {
      console.error('❌ Erreur détection complète:', error);
      
      // Fallback manuel - créer des panels de test
      console.log('🔄 Tentative de fallback manuel...');
      const fallbackPanels = [
        { id: 0, x: 50, y: 50, width: 200, height: 150, confidence: 0.5 },
        { id: 1, x: 300, y: 50, width: 200, height: 150, confidence: 0.5 },
        { id: 2, x: 50, y: 250, width: 200, height: 150, confidence: 0.5 },
        { id: 3, x: 300, y: 250, width: 200, height: 150, confidence: 0.5 }
      ];
      
      setPanels(fallbackPanels);
      console.log('🎯 Panels de fallback définis:', fallbackPanels);
      
    } finally {
      setIsDetecting(false);
    }
  };

  /**
   * Gestionnaire de chargement d'image
   */
  const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;
    setImageSize({ 
      width: img.naturalWidth, 
      height: img.naturalHeight 
    });
    setIsImageLoaded(true);
    console.log(`📷 Image chargée: ${img.naturalWidth}x${img.naturalHeight}`);
  }, []);

  /**
   * Gestionnaire de sélection de panel
   */
  const handlePanelTap = useCallback((panel: Panel, index: number) => {
    console.log(`👆 Panel sélectionné: ${index}`);
    setSelectedPanel(panel);
    setCurrentPanelIndex(index);
    
    // Zoom sur le panel
    zoomToPanel(panel);
    
    if (onPanelSelect) {
      onPanelSelect(panel, index);
    }
  }, [onPanelSelect]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Zoom automatique sur un panel
   */
  const zoomToPanel = useCallback((panel: Panel) => {
    const padding = 40;
    const panelCenterX = panel.x + panel.width / 2;
    const panelCenterY = panel.y + panel.height / 2;

    // Calculer le zoom nécessaire
    const scaleX = (containerSize.width - padding * 2) / panel.width;
    const scaleY = (containerSize.height - padding * 2) / panel.height;
    const targetScale = Math.min(scaleX, scaleY, 3); // Max zoom 3x

    // Calculer la translation pour centrer
    const scaledPanelCenterX = panelCenterX * targetScale;
    const scaledPanelCenterY = panelCenterY * targetScale;

    const targetX = containerSize.width / 2 - scaledPanelCenterX;
    const targetY = containerSize.height / 2 - scaledPanelCenterY;

    // Animation fluide
    scale.set(targetScale);
    x.set(targetX);
    y.set(targetY);
  }, [containerSize, scale, x, y]);

  /**
   * Reset du zoom
   */
  const resetZoom = useCallback(() => {
    scale.set(1);
    x.set(0);
    y.set(0);
    setSelectedPanel(null);
    setCurrentPanelIndex(-1);
  }, [scale, x, y]);

  /**
   * Navigation entre panels
   */
  const navigateToPanel = useCallback((direction: 'next' | 'prev') => {
    if (panels.length === 0) return;
    
    let nextIndex: number;
    if (direction === 'next') {
      nextIndex = (currentPanelIndex + 1) % panels.length;
    } else {
      nextIndex = currentPanelIndex <= 0 ? panels.length - 1 : currentPanelIndex - 1;
    }
    
    const targetPanel = panels[nextIndex];
    handlePanelTap(targetPanel, nextIndex);
  }, [panels, currentPanelIndex, handlePanelTap]);

  /**
   * Gestionnaire de drag/pan
   */
  const handleDrag = useCallback((event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    const currentScale = scale.get();
    x.set(x.get() + info.delta.x / currentScale);
    y.set(y.get() + info.delta.y / currentScale);
  }, [scale, x, y]);

  /**
   * Gestionnaire de wheel pour zoom
   */
  const handleWheel = useCallback((event: React.WheelEvent) => {
    event.preventDefault();
    const currentScale = scale.get();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    const newScale = Math.max(0.5, Math.min(5, currentScale * delta));
    scale.set(newScale);
  }, [scale]);

  /**
   * Gestionnaire de double clic
   */
  const handleDoubleClick = useCallback(() => {
    if (scale.get() > 1) {
      resetZoom();
    } else {
      scale.set(2);
    }
  }, [scale, resetZoom]);

  // Gestion des erreurs
  if (error) {
    return (
      <div className="flex items-center justify-center h-full bg-black text-white">
        <div className="text-center">
          <p className="text-lg mb-2">❌ Erreur</p>
          <p className="text-sm opacity-70">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div 
      ref={containerRef}
      className={`relative w-full h-full overflow-hidden bg-black ${className}`}
      onWheel={handleWheel}
    >
      {/* Image avec gestes */}
      <motion.div
        style={{ transform }}
        drag
        onDrag={handleDrag}
        dragConstraints={false}
        dragElastic={0.1}
        onDoubleClick={handleDoubleClick}
        className="cursor-move relative"
      >
        <Image
          ref={imageRef}
          src={imageUri}
          alt="Comics page"
          width={imageSize.width || 800}
          height={imageSize.height || 600}
          className="max-w-none select-none"
          onLoad={handleImageLoad}
          priority
        />
        
        {/* Overlay des panels */}
        {showPanels && panels.length > 0 && (
          <PanelOverlay
            panels={panels}
            selectedPanel={selectedPanel}
            onPanelPress={handlePanelTap}
            imageSize={imageSize}
          />
        )}
      </motion.div>

      {/* Contrôles de navigation */}
      {panels.length > 1 && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex space-x-2">
          <motion.button
            className="bg-black/70 text-white px-4 py-2 rounded-full hover:bg-black/90 transition-colors"
            onClick={() => navigateToPanel('prev')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Précédent
          </motion.button>
          <motion.button
            className="bg-black/70 text-white px-4 py-2 rounded-full hover:bg-black/90 transition-colors"
            onClick={() => navigateToPanel('next')}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Suivant →
          </motion.button>
        </div>
      )}

      {/* Bouton de reset du zoom */}
      {scale.get() > 1 && (
        <motion.button
          className="absolute top-4 right-4 bg-black/70 text-white p-2 rounded-full hover:bg-black/90 transition-colors"
          onClick={resetZoom}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          🔄
        </motion.button>
      )}

      {/* Indicateur de chargement */}
      <LoadingIndicator
        visible={isDetecting}
        isLoading={isDetecting}
        text="Détection IA en cours..."
        subText="Analyse des panels de la BD..."
      />

      {/* Debug info en développement */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute top-4 left-4 bg-black/80 text-white p-3 rounded text-sm max-w-xs">
          <p className="font-bold text-green-400">Debug Info:</p>
          <p>Panels: {panels.length}</p>
          <p>ShowPanels: {showPanels ? 'OUI' : 'NON'}</p>
          <p>AutoDetect: {autoDetect ? 'OUI' : 'NON'}</p>
          <p>IsDetecting: {isDetecting ? 'OUI' : 'NON'}</p>
          <p>ImageLoaded: {isImageLoaded ? 'OUI' : 'NON'}</p>
          <p>ImageSize: {imageSize.width}x{imageSize.height}</p>
          <p>Panel actuel: {currentPanelIndex + 1}</p>
          <p>Zoom: {scale.get().toFixed(1)}x</p>
          {panels.length > 0 && (
            <div className="mt-2 text-xs">
              <p className="text-yellow-400">Premiers panels:</p>
              {panels.slice(0, 2).map(panel => (
                <p key={panel.id}>#{panel.id}: {Math.round(panel.x)},{Math.round(panel.y)} - {Math.round(panel.width)}x{Math.round(panel.height)}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ComicsViewer;