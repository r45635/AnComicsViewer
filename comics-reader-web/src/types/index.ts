/**
 * Types pour l'application Comics Reader Web
 */

export interface Panel {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number;
}

export interface ComicsItem {
  id: string;
  title: string;
  type: 'pdf' | 'image';
  originalUri: string;
  pages: string[];
  pageCount: number;
  addedDate: string;
  thumbnail?: string;
}

export interface ImageSize {
  width: number;
  height: number;
}

export interface PanelDetectionResult {
  panels: Panel[];
  processingTime?: number;
  modelVersion?: string;
}

export interface LoadingState {
  isLoading: boolean;
  text?: string;
  subText?: string;
  progress?: number;
}

export interface ViewerSettings {
  showPanels: boolean;
  autoDetect: boolean;
  panelNavigation: boolean;
  zoomSensitivity: number;
}

export type NavigationDirection = 'next' | 'prev' | 'first' | 'last';

export interface ZoomSettings {
  scale: number;
  translateX: number;
  translateY: number;
  minScale: number;
  maxScale: number;
}