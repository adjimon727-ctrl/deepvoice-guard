import os
import wave
import numpy as np
import librosa
from PIL import Image
from transformers import pipeline

# 1. Chargement du modèle de classification d'images depuis Hugging Face
print("Chargement du modèle de détection de deepfake audio...")
model_id = "kubinooo/convnext-tiny-224-audio-deepfake-classification"
classifier = pipeline("image-classification", model=model_id)

def audio_to_mel_spectrogram(audio_path, duration=2.0):
    """
    Charge un fichier audio, extrait les 2 premières secondes et 
    le convertit en une image de mel-spectrogramme compatible avec le modèle.
    """
    print("Lecture du fichier audio...")
    try:
        # Essai de lecture native ultra-robuste avec le module wave
        with wave.open(audio_path, 'rb') as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            
            # Conversion des données binaires en tableau lisible par l'IA
            if sampwidth == 2:
                y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                y = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                raise ValueError("Format audio non supporté, utilise un WAV 16-bit standard.")
                
            # Si le fichier est en stéréo, on le convertit en mono (requis pour l'IA)
            if channels > 1:
                y = y.reshape(-1, channels)[:, 0]
                
            # Ajustement de la durée demandée
            if duration:
                y = y[:int(sr * duration)]
                
    except Exception as e:
        # Gilet de sauvetage : Si le WAV a un encodage bizarroïde, librosa tente de prendre le relais
        print(f"Note : Lecture alternative ({e}).")
        y, sr = librosa.load(audio_path, duration=duration)
    
    # Si l'audio est plus court que 2 secondes, on rajoute du vide (padding)
    target_length = int(duration * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')

    # Générer le mel-spectrogramme (taille standard de 224 pour ConvNeXt)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
    
    # Convertir en échelle décibel (dB)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Normaliser les valeurs entre 0 et 255 pour créer une image
    S_db_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min()) * 255
    img_array = S_db_norm.astype(np.uint8)
    
    # Inverser l'axe vertical pour que les basses fréquences soient en bas
    img_array = np.flipud(img_array)
    
    # Convertir le tableau numpy en image PIL au format RGB (attendu par ConvNeXt)
    image = Image.fromarray(img_array).convert("RGB")
    return image

def predict_audio(audio_path):
    """
    Effectue la prédiction pour savoir si l'audio est réel ou un deepfake.
    """
    if not os.path.exists(audio_path):
        print(f"Erreur : Le fichier '{audio_path}' est introuvable.")
        return

    print(f"\nAnalyse du fichier : {audio_path}...")
    
    try:
        # Étape de conversion de l'audio en image
        img_spectrogram = audio_to_mel_spectrogram(audio_path)
        
        # Étape de classification par l'IA
        predictions = classifier(img_spectrogram)
        
        # Affichage des résultats
        print("\n--- RÉSULTATS DE L'ANALYSE ---")
        for pred in predictions:
            score_pourcentage = pred['score'] * 100
            label = pred['label']
            print(f"• {label} : {score_pourcentage:.2f}%")
            
    except Exception as e:
        print(f"Une erreur est survenue lors du traitement : {e}")

# --- ZONE DE TEST ---
if __name__ == "__main__":
    # Nom de ton fichier de test
    fichier_audio = "mon_test.wav" 
    
    # Lancement de la détection
    predict_audio(fichier_audio)