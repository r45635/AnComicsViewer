'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Panel, ImageSize } from '@/types';

interface PanelOverlayProps {
  panels: Panel[];
  selectedPanel?: Panel | null;
  onPanelPress: (panel: Panel, index: number) => void;
  imageSize: ImageSize;
  className?: string;
}

export const PanelOverlay: React.FC<PanelOverlayProps> = ({
  panels,
  selectedPanel,
  onPanelPress,
  imageSize,
  className = ''
}) => {
  console.log('🎨 PanelOverlay render:', { panels: panels?.length, imageSize, className });
  
  if (!panels || panels.length === 0) {
    console.log('⚠️ PanelOverlay: Aucun panel à afficher');
    return null;
  }

  return (
    <div 
      className={`absolute inset-0 pointer-events-none ${className}`}
      style={{
        width: imageSize.width,
        height: imageSize.height
      }}
    >
      <AnimatePresence>
        {panels.map((panel, index) => {
          const isSelected = selectedPanel?.id === panel.id;
          
          return (
            <motion.div
              key={panel.id}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              transition={{ delay: index * 0.1 }}
              className="absolute pointer-events-auto cursor-pointer group"
              style={{
                left: panel.x,
                top: panel.y,
                width: panel.width,
                height: panel.height,
              }}
              onClick={() => onPanelPress(panel, index)}
            >
              {/* Bordure du panel */}
              <motion.div
                className={`
                  absolute inset-0 border-2 rounded-md transition-all duration-200
                  ${isSelected 
                    ? 'border-blue-500 bg-blue-500/20 shadow-lg shadow-blue-500/25' 
                    : 'border-green-400 bg-green-400/10 hover:border-green-300 hover:bg-green-400/20'
                  }
                `}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              />

              {/* Numéro du panel */}
              <motion.div
                className={`
                  absolute -top-2 -left-2 w-6 h-6 rounded-full text-xs font-bold
                  flex items-center justify-center text-white shadow-lg
                  ${isSelected ? 'bg-blue-500' : 'bg-green-500'}
                `}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: index * 0.1 + 0.2 }}
              >
                {index + 1}
              </motion.div>

              {/* Indicateur de confiance (si disponible) */}
              {panel.confidence && (
                <motion.div
                  className="absolute -bottom-2 -right-2 px-2 py-1 bg-black/70 text-white text-xs rounded-md"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: index * 0.1 + 0.3 }}
                >
                  {Math.round(panel.confidence * 100)}%
                </motion.div>
              )}

              {/* Animation de pulsation pour le panel sélectionné */}
              {isSelected && (
                <motion.div
                  className="absolute inset-0 border-2 border-blue-400 rounded-md"
                  animate={{
                    opacity: [0.5, 1, 0.5],
                    scale: [1, 1.02, 1]
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                />
              )}

              {/* Zone de hover pour une meilleure UX */}
              <div className="absolute inset-0 bg-transparent group-hover:bg-white/5 rounded-md transition-colors duration-200" />
            </motion.div>
          );
        })}
      </AnimatePresence>

      {/* Indicateur du nombre total de panels */}
      {panels.length > 0 && (
        <motion.div
          className="absolute top-4 right-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm font-medium"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          {panels.length} panel{panels.length > 1 ? 's' : ''}
        </motion.div>
      )}
    </div>
  );
};

export default PanelOverlay;