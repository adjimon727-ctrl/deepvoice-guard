import os

# Ajoute ta clé Hugging Face ici (elle commence par hf_...)
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
import wave
import shutil
import numpy as np
import librosa
import cv2  # Pour le traitement vidéo
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# 1. Initialisation de l'API
app = FastAPI(title="🛡️ DeepVoice Guard - Suite Cyber Multimodale")

# 2. Chargement des modèles au démarrage
print("Chargement des modèles d'IA...")
# Modèle Audio
audio_model_id = "kubinooo/convnext-tiny-224-audio-deepfake-classification"
audio_classifier = pipeline("image-classification", model=audio_model_id)

# Modèle Image (Chargement local depuis ton dossier)
image_model_id = "kubinooo/convnext-tiny-224-audio-deepfake-classification"
image_classifier = pipeline("image-classification", model=image_model_id)

def audio_to_mel_spectrogram(audio_path, duration=2.0):
    """Convertit l'audio en spectrogramme pour l'analyse"""
    try:
        with wave.open(audio_path, 'rb') as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            
            if sampwidth == 2:
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                y = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise ValueError("Format WAV non supporté.")
            if channels > 1:
                y = y.reshape(-1, channels)[:, 0]
            if duration:
                y = y[:int(sr * duration)]
    except Exception:
        y, sr = librosa.load(audio_path, duration=duration)
    
    target_length = int(duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
    img_array = S_db_norm.astype(np.uint8)
    img_array = np.flipud(img_array)
    return Image.fromarray(img_array).convert("RGB")


def analyze_video_frames(video_path, max_frames=5):
    """Extrait des images clés de la vidéo et fait la moyenne des scores d'analyse"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        raise ValueError("Fichier vidéo corrompu ou illisible.")
        
    frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
    fake_scores = []
    real_scores = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        predictions = image_classifier(pil_img)
        
        res = {}
        for pred in predictions:
            lbl = pred['label'].lower()
            if lbl in ['fake', 'ai', 'artificial']:
                res['fake'] = pred['score'] * 100
            elif lbl in ['real', 'human', 'authentic']:
                res['real'] = pred['score'] * 100
                
        fake_scores.append(res.get('fake', 0))
        real_scores.append(res.get('real', 0))
        
    cap.release()
    if not fake_scores:
        raise ValueError("Aucune image n'a pu être extraite de la vidéo.")
        
    return {
        "fake": round(float(np.mean(fake_scores)), 2),
        "real": round(float(np.mean(real_scores)), 2)
    }


# 3. INTERFACE WEB MULTIMODALE
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepVoice Guard - Suite Cyber Multimodale</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#0f172a] text-[#f8fafc] flex flex-col items-center justify-start min-h-screen p-6 space-y-8">
    
    <!-- Conteneur Principal -->
    <div class="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl mt-4">
        
        <!-- Header -->
        <div class="text-center mb-6">
            <span class="bg-indigo-500/10 text-indigo-400 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider border border-indigo-500/20">Suite de Sécurité B2B</span>
            <h1 class="text-3xl font-extrabold mt-3 tracking-tight">🛡️ DeepVoice <span class="text-indigo-500">Guard</span></h1>
            <p class="text-slate-400 mt-2 text-sm">Détection de falsifications par IA sur les fichiers Audios, Images et Vidéos.</p>
        </div>

        <!-- Système d'onglets pour le sélecteur de mode -->
        <div class="flex bg-slate-950 p-1.5 rounded-xl border border-slate-800 mb-6 text-sm font-medium">
            <button id="tab-audio" onclick="setMode('audio')" class="flex-1 py-2.5 rounded-lg text-center transition-all bg-indigo-600 text-white shadow-sm">🎵 Audio</button>
            <button id="tab-image" onclick="setMode('image')" class="flex-1 py-2.5 rounded-lg text-center transition-all text-slate-400 hover:text-white">🖼️ Image</button>
            <button id="tab-video" onclick="setMode('video')" class="flex-1 py-2.5 rounded-lg text-center transition-all text-slate-400 hover:text-white">📹 Vidéo</button>
        </div>

        <!-- Formulaire de dépôt -->
        <div id="dropzone" class="border-2 border-dashed border-slate-700 hover:border-indigo-500 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-950/40">
            <input type="file" id="mediaFile" accept="audio/*" class="hidden">
            <div class="space-y-2">
                <p class="text-base font-medium">Glissez-déposez votre fichier <span id="text-mode-file">audio</span> ici</p>
                <p class="text-xs text-slate-500" id="text-formats">Formats supportés : WAV, MP3, M4A</p>
                <button type="button" class="mt-4 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-colors">Parcourir</button>
            </div>
        </div>

        <!-- Fichier sélectionné -->
        <div id="fileInfo" class="hidden mt-4 p-3 bg-slate-800/50 rounded-lg flex items-center justify-between text-xs text-slate-300">
            <span id="fileName" class="font-mono truncate max-w-[400px]">fichier.ext</span>
            <span class="text-indigo-400 font-semibold">Prêt pour l'analyse</span>
        </div>

        <!-- Bouton d'action -->
        <button id="btnAnalyze" class="w-full mt-6 bg-slate-800 text-slate-500 font-bold py-3 px-4 rounded-xl cursor-not-allowed transition-all" disabled>
            Lancer l'analyse biométrique
        </button>

        <!-- Résultats -->
        <div id="resultSection" class="hidden mt-8 pt-6 border-t border-slate-800">
            <div class="flex items-center justify-between mb-6">
                <h3 class="text-lg font-bold">📊 Résultat de l'analyse :</h3>
                <span id="mainVerdict" class="uppercase text-xs font-extrabold px-3 py-1 rounded-full"></span>
            </div>
            
            <div class="space-y-4">
                <div>
                    <div class="flex justify-between text-xs mb-1 font-medium text-slate-400">
                        <span>Indice de falsification / Contrefaçon (FAKE)</span>
                        <span id="fakeScore" class="font-bold text-red-400">0%</span>
                    </div>
                    <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                        <div id="fakeBar" class="bg-red-500 h-3 rounded-full transition-all duration-500" style="width: 0%"></div>
                    </div>
                </div>
                <div>
                    <div class="flex justify-between text-xs mb-1 font-medium text-slate-400">
                        <span>Fidélité de l'élément d'origine (REAL)</span>
                        <span id="realScore" class="font-bold text-emerald-400">0%</span>
                    </div>
                    <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden">
                        <div id="realBar" class="bg-emerald-500 h-3 rounded-full transition-all duration-500" style="width: 0%"></div>
                    </div>
                </div>
            </div>

            <!-- Rapport PDF -->
            <div class="mt-6 flex justify-end">
                <button id="btnPrintReport" class="bg-slate-800 hover:bg-slate-750 text-slate-300 hover:text-white text-xs font-semibold px-4 py-2 rounded-lg border border-slate-700 transition-all flex items-center gap-2">
                    📄 Télécharger le rapport officiel (PDF)
                </button>
            </div>
        </div>
    </div>

    <!-- Section Historique -->
    <div class="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl mb-8">
        <div class="flex justify-between items-center mb-4">
            <h3 class="text-base font-bold flex items-center gap-2">🕒 Historique global des audits</h3>
            <button id="btnClearHistory" class="text-xs text-slate-500 hover:text-red-400 transition-colors">Effacer tout</button>
        </div>
        <div class="overflow-x-auto">
            <table class="w-full text-left text-xs text-slate-400">
                <thead class="bg-slate-950 text-slate-400 uppercase font-semibold border-b border-slate-800">
                    <tr>
                        <th class="p-3">Type</th>
                        <th class="p-3">Nom du Fichier</th>
                        <th class="p-3">Verdict</th>
                        <th class="p-3 text-right">Score Fake</th>
                    </tr>
                </thead>
                <tbody id="historyTableBody"></tbody>
            </table>
        </div>
    </div>

    <script>
        let currentMode = 'audio';
        let lastScanData = null;

        const fileInput = document.getElementById('mediaFile');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const btnAnalyze = document.getElementById('btnAnalyze');
        const resultSection = document.getElementById('resultSection');
        const historyTableBody = document.getElementById('historyTableBody');

        document.addEventListener('DOMContentLoaded', displayHistory);
        document.getElementById('dropzone').addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);

        document.getElementById('btnClearHistory').addEventListener('click', () => {
            if (confirm("Voulez-vous vraiment vider tout l'historique des audits ?")) {
                localStorage.removeItem('multimodal_history');
                displayHistory();
            }
        });

        function setMode(mode) {
            currentMode = mode;
            
            fileInput.value = '';
            fileInfo.classList.add('hidden');
            btnAnalyze.setAttribute('disabled', 'true');
            btnAnalyze.className = "w-full mt-6 bg-slate-800 text-slate-500 font-bold py-3 px-4 rounded-xl cursor-not-allowed transition-all";

            ['audio', 'image', 'video'].forEach(m => {
                const btn = document.getElementById('tab-' + m);
                if(m === mode) {
                    btn.className = "flex-1 py-2.5 rounded-lg text-center transition-all bg-indigo-600 text-white shadow-sm";
                } else {
                    btn.className = "flex-1 py-2.5 rounded-lg text-center transition-all text-slate-400 hover:text-white";
                }
            });

            if (mode === 'audio') {
                fileInput.setAttribute('accept', 'audio/*');
                document.getElementById('text-mode-file').textContent = 'audio';
                document.getElementById('text-formats').textContent = 'Formats supportés : WAV, MP3, M4A';
            } else if (mode === 'image') {
                fileInput.setAttribute('accept', 'image/*');
                document.getElementById('text-mode-file').textContent = 'image';
                document.getElementById('text-formats').textContent = 'Formats supportés : JPG, PNG, JPEG';
            } else if (mode === 'video') {
                fileInput.setAttribute('accept', 'video/*');
                document.getElementById('text-mode-file').textContent = 'vidéo';
                document.getElementById('text-formats').textContent = 'Formats supportés : MP4, AVI, MOV';
            }
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) {
                fileName.textContent = file.name;
                fileInfo.classList.remove('hidden');
                btnAnalyze.removeAttribute('disabled');
                btnAnalyze.className = "w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg cursor-pointer transition-all text-center";
            }
        }

        btnAnalyze.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) return;

            if (currentMode === 'audio' || currentMode === 'video') {
                alert("Fonctionnalite Premium\\n\\nL'analyse avancee des flux Audios et Videos est reservee aux abonnes Premium et infrastructures B2B.\\n\\nLa version de demonstration gratuite est limitee a l'analyse d'Images.\\n\\nPour deployer DeepVoice Guard au sein de votre entreprise, contactez-nous.");
                return;
            }

            btnAnalyze.textContent = "Analyse algorithmique en cours...";
            btnAnalyze.className = "w-full mt-6 bg-slate-700 text-slate-400 font-bold py-3 px-4 rounded-xl cursor-wait text-center";
            
            const formData = new FormData();
            formData.append("file", file);
            formData.append("mode", currentMode);

            try {
                const response = await fetch('/scan', { method: 'POST', body: formData });
                const data = await response.json();

                if (response.ok) {
                    resultSection.classList.remove('hidden');
                    
                    const fake = data.analyses.fake || 0;
                    const real = data.analyses.real || 0;

                    document.getElementById('fakeScore').textContent = fake + '%';
                    document.getElementById('fakeBar').style.width = fake + '%';
                    document.getElementById('realScore').textContent = real + '%';
                    document.getElementById('realBar').style.width = real + '%';

                    const verdictBadge = document.getElementById('mainVerdict');
                    let verdictText = '';
                    if (data.verdict_principal === 'fake') {
                        verdictText = 'Element Contrefait / Fake';
                        verdictBadge.className = 'uppercase text-xs font-extrabold px-3 py-1 rounded-full bg-red-500/10 text-red-400 border border-red-500/20';
                    } else {
                        verdictText = 'Element Authentique / Real';
                        verdictBadge.className = 'uppercase text-xs font-extrabold px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
                    }
                    verdictBadge.textContent = verdictText;

                    lastScanData = { type: currentMode.toUpperCase(), name: file.name, verdict: verdictText, fake, real };
                    saveToHistory(currentMode.toUpperCase(), file.name, verdictText, fake);
                } else {
                    alert("Erreur : " + data.detail);
                }
            } catch (err) {
                alert("Erreur de connexion au serveur.");
            } finally {
                btnAnalyze.textContent = "Lancer l'analyse biometrique";
                btnAnalyze.className = "w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg cursor-pointer transition-all text-center";
            }
        });

        function saveToHistory(type, name, verdict, fakeScore) {
            let history = JSON.parse(localStorage.getItem('multimodal_history')) || [];
            history.unshift({ type, name, verdict, fakeScore });
            if (history.length > 6) history.pop();
            localStorage.setItem('multimodal_history', JSON.stringify(history));
            displayHistory();
        }

        function displayHistory() {
            let history = JSON.parse(localStorage.getItem('multimodal_history')) || [];
            historyTableBody.innerHTML = '';
            
            if (history.length === 0) {
                historyTableBody.innerHTML = `<tr><td colspan="4" class="p-4 text-center text-slate-500 italic">Aucune analyse récente</td></tr>`;
                return;
            }

            history.forEach(item => {
                const isFake = item.verdict.includes('Fake');
                const badgeClass = isFake ? 'text-red-400 bg-red-500/10 border border-red-500/20' : 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20';
                
                let icon = '🎵';
                if(item.type === 'IMAGE') icon = '🖼️';
                if(item.type === 'VIDEO') icon = '📹';

                const row = document.createElement('tr');
                row.className = 'border-b border-slate-800/60 hover:bg-slate-800/20 transition-colors';
                row.innerHTML = `
                    <td class="p-3 text-base">${icon}</td>
                    <td class="p-3 font-mono text-[11px] max-w-[200px] truncate text-slate-300">${item.name}</td>
                    <td class="p-3"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${badgeClass}">${item.verdict}</span></td>
                    <td class="p-3 text-right font-semibold ${isFake ? 'text-red-400' : 'text-emerald-400'}">${item.fakeScore}%</td>
                `;
                historyTableBody.appendChild(row);
            });
        }

        document.getElementById('btnPrintReport').addEventListener('click', () => {
            if (!lastScanData) return;
            const printWindow = window.open('', '_blank');
            const dateString = new Date().toLocaleString('fr-FR');
            
            printWindow.document.write(`
                <html>
                <head>
                    <title>Rapport d'Audit Cyber - DeepVoice Guard</title>
                    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
                    <style>@media print { body { -webkit-print-color-adjust: exact; } }</style>
                </head>
                <body class="bg-white text-slate-900 p-12 font-sans">
                    <div class="max-w-3xl mx-auto border-4 border-slate-900 p-8 rounded-xl">
                        <div class="flex justify-between items-center border-b-2 border-slate-900 pb-6 mb-8">
                            <div>
                                <h1 class="text-2xl font-black uppercase tracking-tight">🛡️ DeepVoice Guard Suite</h1>
                                <p class="text-xs text-slate-500 uppercase tracking-widest font-semibold mt-1">Rapport d'Expertise Médias Multimodal</p>
                            </div>
                            <div class="text-right text-xs text-slate-500">
                                <p><strong>Date d'analyse :</strong> ${dateString}</p>
                                <p><strong>Type de flux :</strong> ${lastScanData.type}</p>
                            </div>
                        </div>
                        
                        <h2 class="text-xl font-bold text-slate-800 mb-4">Analyse de Contrefaçon Numérique</h2>
                        
                        <div class="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-6 font-mono text-xs space-y-1">
                            <p><strong>Fichier examiné :</strong> ${lastScanData.name}</p>
                            <p><strong>Vérification :</strong> Intégrité structurelle des métadonnées et pixels/fréquences</p>
                        </div>

                        <div class="border border-slate-200 rounded-xl p-6 text-center mb-8 bg-slate-50/50">
                            <p class="text-sm uppercase tracking-wider font-semibold text-slate-500">Verdict de l'Intelligence Artificielle</p>
                            <p class="text-2xl font-black mt-2 uppercase tracking-tight ${lastScanData.verdict.includes('Fake') ? 'text-red-600' : 'text-emerald-600'}">
                                ${lastScanData.verdict}
                            </p>
                        </div>

                        <h3 class="font-bold text-sm text-slate-700 uppercase tracking-wider mb-3">Résultats de la Télémétrie</h3>
                        <div class="space-y-4 border border-slate-200 rounded-xl p-4 bg-white mb-8">
                            <div>
                                <div class="flex justify-between text-xs mb-1 font-semibold">
                                    <span>Probabilité d'Anomalie/Génération Synthétique (FAKE)</span>
                                    <span class="text-red-600">${lastScanData.fake}%</span>
                                </div>
                                <div class="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                                    <div class="bg-red-600 h-3" style="width: ${lastScanData.fake}%"></div>
                                </div>
                            </div>
                            <div>
                                <div class="flex justify-between text-xs mb-1 font-semibold">
                                    <span>Indice d'Élément Réel / Captation Physique (REAL)</span>
                                    <span class="text-emerald-600">${lastScanData.real}%</span>
                                </div>
                                <div class="w-full bg-slate-200 rounded-full h-3 overflow-hidden">
                                    <div class="bg-emerald-600 h-3" style="width: ${lastScanData.real}%"></div>
                                </div>
                            </div>
                        </div>

                        <div class="border-t border-slate-200 pt-6 flex justify-between items-center text-[10px] text-slate-400 font-mono">
                            <p>ID Vérification : DVG-MULTIMODAL-${lastScanData.type}-SECURE</p>
                            <p>© ${new Date().getFullYear()} - Document d'audit confidentiel</p>
                        </div>
                    </div>
                </body>
                </html>
            `);
            printWindow.document.close();
            setTimeout(() => {
                printWindow.print();
            }, 500);
        });
    </script>
</body>
</html>
    """


# 4. ROUTE DE DÉTECTION MULTIMODALE UNIQUE
@app.post("/scan")
async def scan_media(file: UploadFile = File(...), mode: str = Form(...)):
    filename_lower = file.filename.lower()
    
    # Validation du format selon le mode
    if mode == "audio" and not filename_lower.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Veuillez envoyer un fichier audio valide (.wav, .mp3, .m4a).")
    elif mode == "image" and not filename_lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une image valide (.jpg, .png, .webp).")
    elif mode == "video" and not filename_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Veuillez envoyer une vidéo valide (.mp4, .avi, .mov).")

    # Écriture du fichier temporaire
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        resultats = {"fake": 0.0, "real": 0.0}
        
        # LOGIQUE 1 : AUDIO
        if mode == "audio":
            img_spectrogram = audio_to_mel_spectrogram(temp_path)
            predictions = audio_classifier(img_spectrogram)
            for pred in predictions:
                lbl = pred['label'].lower()
                if lbl in ['fake', 'ai', 'artificial', 'synthetic', 'spoof']:
                    resultats['fake'] = round(pred['score'] * 100, 2)
                elif lbl in ['real', 'human', 'authentic']:
                    resultats['real'] = round(pred['score'] * 100, 2)
                else:
                    resultats[lbl] = round(pred['score'] * 100, 2)
                
        # LOGIQUE 2 : IMAGE
        elif mode == "image":
            pil_img = Image.open(temp_path).convert("RGB")
            predictions = image_classifier(pil_img)
            for pred in predictions:
                lbl = pred['label'].lower()
                if lbl in ['fake', 'ai', 'artificial', 'synthetic']:
                    resultats['fake'] = round(pred['score'] * 100, 2)
                elif lbl in ['real', 'human', 'authentic']:
                    resultats['real'] = round(pred['score'] * 100, 2)
                else:
                    resultats[lbl] = round(pred['score'] * 100, 2)
                
        # LOGIQUE 3 : VIDÉO
        elif mode == "video":
            resultats = analyze_video_frames(temp_path, max_frames=5)

        # Calcul du verdict global
        verdict = "fake" if resultats.get("fake", 0) > resultats.get("real", 0) else "real"
        
        return {
            "status": "success",
            "filename": file.filename,
            "mode": mode,
            "verdict_principal": verdict,
            "analyses": resultats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse : {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)