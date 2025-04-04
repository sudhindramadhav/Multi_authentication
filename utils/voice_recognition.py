#voice_recognition.py

import os
import speech_recognition as sr
import numpy as np
import librosa
import base64
import shutil
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler

DATA_FOLDER = 'data'
os.makedirs(DATA_FOLDER, exist_ok=True)

def capture_voice_data():
    """Improved voice capture with better error handling"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    temp_file_path = os.path.join(DATA_FOLDER, 'temp.wav')

    try:
        with mic as source:
            print("🎤 Adjusting for ambient noise (please wait)...")
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Longer adjustment
            print("🔊 Please say the passphrase clearly now!")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                print("❌ No audio detected - please speak when prompted")
                return None, None

        # Save audio
        with open(temp_file_path, 'wb') as f:
            f.write(audio.get_wav_data())

        # Verify audio is valid
        if os.path.getsize(temp_file_path) < 1024:  # Check minimum file size
            print("⚠️ Audio capture too short")
            return None, None

        # Get base64 data
        with open(temp_file_path, 'rb') as f:
            voice_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Convert speech to text
        try:
            spoken_text = recognizer.recognize_google(audio).lower()
            print(f"✅ Captured phrase: '{spoken_text}'")
            return voice_data, spoken_text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return voice_data, None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return voice_data, None
            
    except Exception as e:
        print(f"❌ Error capturing voice: {str(e)}")
        return None, None

def extract_voice_features(file_path):
    """Extracts MFCC and chroma features from voice audio."""
    try:
        y, sr = librosa.load(file_path, sr=None)

        if y is None or len(y) < sr * 0.5:  # Ensure at least 0.5 sec of audio
            print(f"⚠️ Error: Audio too short in {file_path}")
            return None

        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        if mfccs is None or chroma is None:
            print(f"⚠️ Feature extraction failed for {file_path}")
            return None

        features = np.hstack((mfccs, chroma))
        return features
    except Exception as e:
        print(f"❌ Error extracting features from {file_path}: {e}")
        return None

def calculate_voice_similarity(stored_voice_path, new_voice_data, stored_text, spoken_text):
    """Improved voice similarity calculation"""
    new_voice_file = os.path.join(DATA_FOLDER, 'temp_new.wav')
    
    try:
        # Save new voice data to file
        with open(new_voice_file, 'wb') as f:
            f.write(base64.b64decode(new_voice_data))

        # Extract features
        stored_features = extract_voice_features(stored_voice_path)
        new_features = extract_voice_features(new_voice_file)

        if stored_features is None or new_features is None:
            print("⚠️ Voice features not extracted properly!")
            return 0.0, False

        # Ensure features are 1-D arrays
        stored_features = np.asarray(stored_features).flatten()
        new_features = np.asarray(new_features).flatten()

        # Handle case where arrays might be different lengths
        min_length = min(len(stored_features), len(new_features))
        if min_length == 0:
            return 0.0, False
            
        stored_features = stored_features[:min_length]
        new_features = new_features[:min_length]

        # Calculate DTW distance
        try:
            distance, _ = fastdtw(
                stored_features.reshape(-1, 1), 
                new_features.reshape(-1, 1), 
                dist=euclidean
            )
            
            # Normalize distance to similarity score
            max_possible_distance = np.sqrt(min_length * (np.max(stored_features)**2 + np.max(new_features)**2))
            similarity = 1 - (distance / max_possible_distance) if max_possible_distance > 0 else 0
            
            # Text comparison
            text_match = (stored_text is not None and 
                          spoken_text is not None and 
                          stored_text.strip().lower() == spoken_text.strip().lower())

            return max(0.0, min(similarity, 1.0)), text_match
            
        except Exception as e:
            print(f"❌ DTW calculation error: {e}")
            return 0.0, False
            
    except Exception as e:
        print(f"❌ Error in similarity calculation: {e}")
        return 0.0, False
    finally:
        try:
            os.remove(new_voice_file)
        except:
            pass
