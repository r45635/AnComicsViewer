'use client';

import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Progress } from '@/components/ui/progress';
import { LoadingState } from '@/types';

interface LoadingIndicatorProps extends LoadingState {
  visible: boolean;
}

export const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({
  visible,
  isLoading,
  text = 'Chargement...',
  subText,
  progress
}) => {
  return (
    <AnimatePresence>
      {visible && isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ type: "spring", duration: 0.5 }}
            className="bg-white dark:bg-gray-900 rounded-lg p-8 shadow-2xl max-w-sm w-full mx-4"
          >
            {/* Animation du robot IA */}
            <div className="flex justify-center mb-6">
              <motion.div
                animate={{ 
                  rotateY: [0, 180, 360],
                  scale: [1, 1.1, 1]
                }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut"
                }}
                className="text-4xl"
              >
                🤖
              </motion.div>
            </div>

            {/* Texte principal */}
            <h3 className="text-lg font-semibold text-center text-gray-900 dark:text-white mb-2">
              {text}
            </h3>

            {/* Sous-texte */}
            {subText && (
              <p className="text-sm text-gray-600 dark:text-gray-400 text-center mb-4">
                {subText}
              </p>
            )}

            {/* Barre de progression */}
            {progress !== undefined ? (
              <div className="space-y-2">
                <Progress value={progress} className="w-full" />
                <p className="text-xs text-gray-500 text-center">
                  {Math.round(progress)}%
                </p>
              </div>
            ) : (
              /* Spinner animé */
              <div className="flex justify-center">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ 
                    duration: 1,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  className="w-8 h-8 border-3 border-blue-500 border-t-transparent rounded-full"
                />
              </div>
            )}

            {/* Points d'animation pour indiquer le traitement */}
            <div className="flex justify-center mt-4 space-x-1">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 1, 0.5]
                  }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    delay: i * 0.2
                  }}
                  className="w-2 h-2 bg-blue-500 rounded-full"
                />
              ))}
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default LoadingIndicator;