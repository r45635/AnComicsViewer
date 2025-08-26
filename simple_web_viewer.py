#!/usr/bin/env python3
"""
AnComicsViewer - Interface Web Simple
Version simplifiée utilisant directement YOLO
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

import cv2
import numpy as np
from ultralytics import YOLO

class ComicsViewerHandler(BaseHTTPRequestHandler):
    model = None
    
    def init_model(self):
        """Initialise le modèle YOLO si pas encore fait"""
        if self.model is None:
            model_path = "detectors/models/multibd_enhanced_v2.pt"
            print(f"📦 Chargement du modèle: {model_path}")
            self.model = YOLO(model_path)
            print("✅ Modèle chargé avec succès!")
    
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
        .upload-area { border: 2px dashed #007acc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; cursor: pointer; }
        .upload-area:hover { background: #f8f9fa; }
        .upload-area.dragover { background: #e6f3ff; }
        .result { margin: 20px 0; }
        .panel-info { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007acc; }
        #image-result { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .stats { display: flex; justify-content: space-around; background: linear-gradient(135deg, #007acc, #0056b3); color: white; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .stat { text-align: center; }
        .stat h3 { margin: 0; font-size: 2em; }
        .stat p { margin: 5px 0 0 0; opacity: 0.9; }
        .model-info { background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 15px; border-radius: 5px; margin: 10px 0; text-align: center; }
        .loading { text-align: center; padding: 40px; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #007acc; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .error { background: #dc3545; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
        .success { background: #28a745; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 AnComicsViewer Enhanced v2</h1>
            <p>Détection intelligente de panels avec YOLOv8 Multi-BD</p>
            <div class="model-info">
                ✅ Modèle YOLOv8s Multi-BD Enhanced v2 (71% mAP50) - Opérationnel !
            </div>
        </div>
        
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="uploadImage()">
            <h3>📁 Glissez une image de BD ici ou cliquez pour sélectionner</h3>
            <p>Formats supportés: JPG, PNG, BMP | Testé sur: Golden City, Tintin, Pin-up, Sisters</p>
        </div>
        
        <div id="loading" style="display: none;" class="loading">
            <div class="spinner"></div>
            <h3>🤖 Analyse en cours...</h3>
            <p>Détection des panels et balloons avec YOLOv8...</p>
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
                <h3>📋 Détails des détections:</h3>
            </div>
        </div>
        
        <div id="error-message" style="display: none;" class="error">
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
            document.getElementById('error-message').style.display = 'none';
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                showError('Erreur de communication: ' + error);
            });
        }
        
        function showError(message) {
            document.getElementById('error-message').textContent = message;
            document.getElementById('error-message').style.display = 'block';
        }
        
        function displayResults(data) {
            document.getElementById('results').style.display = 'block';
            document.getElementById('panel-count').textContent = data.detection_count;
            document.getElementById('confidence').textContent = data.avg_confidence + '%';
            document.getElementById('processing-time').textContent = data.processing_time + 'ms';
            document.getElementById('image-result').src = 'data:image/jpeg;base64,' + data.result_image;
            
            const detailsDiv = document.getElementById('panel-details');
            detailsDiv.innerHTML = '<h3>📋 Détails des détections:</h3>';
            
            data.detections.forEach((detection, i) => {
                const detectionDiv = document.createElement('div');
                detectionDiv.className = 'panel-info';
                
                const className = detection.class_name === 'panel' ? '🎭 Panel' : '💭 Balloon';
                detectionDiv.innerHTML = `
                    <strong>${className} ${i + 1}:</strong><br>
                    • Type: ${detection.class_name}<br>
                    • Confiance: ${detection.confidence}%<br>
                    • Position: (${detection.x}, ${detection.y}) → (${detection.x2}, ${detection.y2})<br>
                    • Taille: ${detection.width}×${detection.height} px
                `;
                detailsDiv.appendChild(detectionDiv);
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
                # Initialiser le modèle si nécessaire
                self.init_model()
                
                # Lire les données du formulaire
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Sauver l'image temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    # Extraire l'image des données multipart
                    boundary_line = post_data.split(b'\\r\\n')[0]
                    boundary = boundary_line[2:] if boundary_line.startswith(b'--') else boundary_line
                    
                    parts = post_data.split(b'--' + boundary)
                    
                    image_data = None
                    for part in parts:
                        if b'Content-Type: image' in part:
                            header_end = part.find(b'\\r\\n\\r\\n')
                            if header_end != -1:
                                image_data = part[header_end + 4:]
                                # Enlever les \\r\\n de fin
                                if image_data.endswith(b'\\r\\n'):
                                    image_data = image_data[:-2]
                                break
                    
                    if image_data:
                        tmp_file.write(image_data)
                        tmp_file_path = tmp_file.name
                    else:
                        raise Exception("Impossible d'extraire l'image des données multipart")
                
                # Analyser avec YOLO
                start_time = time.time()
                
                # Charger l'image
                image = cv2.imread(tmp_file_path)
                if image is None:
                    raise Exception("Impossible de charger l'image")
                
                # Faire la détection YOLO
                results = self.model(tmp_file_path, conf=0.15, iou=0.6, imgsz=1280, device='mps')
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Traiter les résultats
                detections = []
                total_confidence = 0
                
                if results and len(results) > 0:
                    result = results[0]
                    boxes = result.boxes
                    
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            x1, y1, x2, y2 = xyxy
                            width, height = x2 - x1, y2 - y1
                            
                            class_name = "panel" if cls == 0 else "balloon"
                            
                            detections.append({
                                'x': int(x1), 'y': int(y1), 'x2': int(x2), 'y2': int(y2),
                                'width': int(width), 'height': int(height),
                                'confidence': f"{conf:.1f}",
                                'class_name': class_name
                            })
                            total_confidence += conf
                
                # Créer l'image avec annotations
                result_image = image.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection['x'], detection['y'], detection['x2'], detection['y2']
                    conf = detection['confidence']
                    class_name = detection['class_name']
                    
                    # Couleur selon la classe
                    color = (0, 255, 0) if class_name == 'panel' else (255, 165, 0)  # Vert pour panels, orange pour balloons
                    
                    # Dessiner le rectangle
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                    
                    # Ajouter le texte
                    text = f"{class_name} {conf}%"
                    cv2.putText(result_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Encoder l'image résultat en base64
                _, buffer = cv2.imencode('.jpg', result_image)
                result_image_b64 = base64.b64encode(buffer).decode()
                
                # Préparer la réponse
                avg_confidence = f"{total_confidence / len(detections):.1f}" if detections else "0.0"
                
                response = {
                    'detection_count': len(detections),
                    'avg_confidence': avg_confidence,
                    'processing_time': processing_time,
                    'result_image': result_image_b64,
                    'detections': detections
                }
                
                # Nettoyer le fichier temporaire
                os.unlink(tmp_file_path)
                
                # Envoyer la réponse
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                print(f"❌ Erreur d'analyse: {e}")
                import traceback
                traceback.print_exc()
                
                error_response = {'error': str(e)}
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(error_response).encode())

def main():
    """Lance le serveur web"""
    print("🌐 AnComicsViewer Enhanced v2 - Interface Web")
    print("=" * 55)
    print("🚀 Initialisation du serveur...")
    
    try:
        # Démarrer le serveur
        server = HTTPServer(('localhost', 8080), ComicsViewerHandler)
        
        print("✅ Serveur démarré sur http://localhost:8080")
        print("📖 Ouverture du navigateur...")
        
        # Ouvrir le navigateur
        def open_browser():
            time.sleep(2)  # Attendre que le serveur soit prêt
            webbrowser.open('http://localhost:8080')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        print("🎯 Interface prête ! Glissez une image de BD dans le navigateur")
        print("💡 Utilisez Ctrl+C pour arrêter le serveur")
        server.serve_forever()
        
    except KeyboardInterrupt:
        print("\\n⚠️ Arrêt du serveur...")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
