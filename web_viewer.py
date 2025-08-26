#!/usr/bin/env python3
"""
AnComicsViewer - Interface Web Simple
Contournement pour les problèmes Qt/PySide6 sur macOS
"""

import os
import sys
import base64
import tempfile
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import webbrowser
import threading
import time

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ancomicsviewer.detectors.multibd_detector import MultiBDPanelDetector
import cv2
import numpy as np

class ComicsViewerHandler(BaseHTTPRequestHandler):
    detector = None
    
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html>
<head>
    <title>AnComicsViewer - Interface Web</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
        .header { text-align: center; color: #333; border-bottom: 2px solid #007acc; padding-bottom: 20px; }
        .upload-area { border: 2px dashed #007acc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-area.dragover { background: #e6f3ff; }
        .result { margin: 20px 0; }
        .panel-info { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
        #image-result { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .stats { display: flex; justify-content: space-around; background: #007acc; color: white; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .stat { text-align: center; }
        .model-info { background: #28a745; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 AnComicsViewer Enhanced v2</h1>
            <p>Détection intelligente de panels avec YOLOv8 Multi-BD</p>
            <div class="model-info">
                ✅ Modèle: multibd_enhanced_v2.pt (71% mAP50) - Prêt !
            </div>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            <h3>📁 Glissez une image ici ou cliquez pour sélectionner</h3>
            <p>Formats supportés: JPG, PNG, BMP</p>
        </div>
        
        <div id="loading" style="display: none; text-align: center; padding: 20px;">
            <h3>🤖 Analyse en cours...</h3>
            <p>Détection des panels avec YOLOv8...</p>
        </div>
        
        <div id="results" style="display: none;">
            <div class="stats">
                <div class="stat">
                    <h3 id="panel-count">0</h3>
                    <p>Panels détectés</p>
                </div>
                <div class="stat">
                    <h3 id="confidence">0%</h3>
                    <p>Confiance moyenne</p>
                </div>
                <div class="stat">
                    <h3 id="processing-time">0ms</h3>
                    <p>Temps de traitement</p>
                </div>
            </div>
            
            <div class="result">
                <h3>🖼️ Résultat de la détection:</h3>
                <img id="image-result" src="" alt="Résultat de détection">
            </div>
            
            <div id="panel-details" class="result">
                <h3>📋 Détails des panels:</h3>
            </div>
        </div>
    </div>

    <script>
        function uploadImage() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                displayResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Erreur: ' + error);
            });
        }
        
        function displayResults(data) {
            document.getElementById('results').style.display = 'block';
            document.getElementById('panel-count').textContent = data.panel_count;
            document.getElementById('confidence').textContent = data.avg_confidence + '%';
            document.getElementById('processing-time').textContent = data.processing_time + 'ms';
            document.getElementById('image-result').src = 'data:image/jpeg;base64,' + data.result_image;
            
            const detailsDiv = document.getElementById('panel-details');
            detailsDiv.innerHTML = '<h3>📋 Détails des panels:</h3>';
            
            data.panels.forEach((panel, i) => {
                const panelDiv = document.createElement('div');
                panelDiv.className = 'panel-info';
                panelDiv.innerHTML = `
                    <strong>Panel ${i + 1}:</strong><br>
                    • Confiance: ${panel.confidence}%<br>
                    • Position: (${panel.x}, ${panel.y}) → (${panel.x + panel.w}, ${panel.y + panel.h})<br>
                    • Taille: ${panel.w}×${panel.h} px
                `;
                detailsDiv.appendChild(panelDiv);
            });
        }
        
        // Drag and drop
        const uploadArea = document.querySelector('.upload-area');
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById('fileInput').files = files;
                uploadImage();
            }
        });
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode())
            
        elif self.path == "/analyze":
            self.send_response(405)
            self.end_headers()
    
    def do_POST(self):
        if self.path == "/analyze":
            try:
                # Lire les données du formulaire
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Sauver l'image temporairement
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    # Extraire l'image des données multipart
                    boundary = post_data.split(b'\\r\\n')[0][2:]
                    parts = post_data.split(boundary)
                    
                    image_data = None
                    for part in parts:
                        if b'Content-Type: image' in part:
                            image_start = part.find(b'\\r\\n\\r\\n') + 4
                            image_data = part[image_start:-2]  # Remove trailing \\r\\n
                            break
                    
                    if image_data:
                        tmp_file.write(image_data)
                        tmp_file_path = tmp_file.name
                    else:
                        raise Exception("Impossible d'extraire l'image")
                
                # Analyser avec le détecteur
                start_time = time.time()
                
                if not self.detector:
                    self.detector = MultiBDPanelDetector()
                
                # Charger l'image
                image = cv2.imread(tmp_file_path)
                if image is None:
                    raise Exception("Impossible de charger l'image")
                
                # Détecter les panels
                panels = self.detector.detect_panels(image)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Dessiner les boîtes de détection
                result_image = image.copy()
                panel_data = []
                total_confidence = 0
                
                for panel in panels:
                    x, y, w, h, conf = panel['x'], panel['y'], panel['width'], panel['height'], panel['confidence']
                    
                    # Dessiner le rectangle
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Ajouter le texte de confiance
                    text = f"{conf:.1f}%"
                    cv2.putText(result_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    panel_data.append({
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'confidence': f"{conf:.1f}"
                    })
                    total_confidence += conf
                
                # Encoder l'image résultat en base64
                _, buffer = cv2.imencode('.jpg', result_image)
                result_image_b64 = base64.b64encode(buffer).decode()
                
                # Préparer la réponse
                avg_confidence = f"{total_confidence / len(panels):.1f}" if panels else "0.0"
                
                response = {
                    'panel_count': len(panels),
                    'avg_confidence': avg_confidence,
                    'processing_time': processing_time,
                    'result_image': result_image_b64,
                    'panels': panel_data
                }
                
                # Nettoyer le fichier temporaire
                os.unlink(tmp_file_path)
                
                # Envoyer la réponse
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                error_response = {'error': str(e)}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())

def main():
    """Lance le serveur web"""
    print("🌐 AnComicsViewer - Interface Web")
    print("=" * 50)
    print("🚀 Initialisation du serveur...")
    
    try:
        # Démarrer le serveur
        server = HTTPServer(('localhost', 8080), ComicsViewerHandler)
        
        print("✅ Serveur démarré sur http://localhost:8080")
        print("📖 Ouverture du navigateur...")
        
        # Ouvrir le navigateur
        def open_browser():
            time.sleep(1)  # Attendre que le serveur soit prêt
            webbrowser.open('http://localhost:8080')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🎯 Interface prête ! Utilisez Ctrl+C pour arrêter")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\\n⚠️ Arrêt du serveur...")
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
