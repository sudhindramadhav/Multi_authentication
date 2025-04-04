import os
import logging
import speech_recognition as sr
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
from pymongo import MongoClient
import random
import base64
import io
from utils.pincode import verify_pincode
from utils.encryption import encrypt_data, decrypt_data
from utils.face_recognition import capture_face_data, calculate_similarity
from utils.voice_recognition import capture_voice_data, calculate_voice_similarity
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt

app = Flask(__name__)

similarity_scores = []

# Set up logging
log_directory = 'log'
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file = os.path.join(log_directory, 'app.log')

logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['fyp2']
collection = db['fyp2_details']

def generate_unique_user_id():
    while True:
        user_id = str(random.randint(1000000, 9999999))  # Generate a 7-digit user ID
        if not collection.find_one({'user_id': user_id}):
            return user_id

def create_log_entry(event, user_id, details=""):
    log_message = f"Event: {event}, User ID: {user_id}, Details: {details}"
    logging.info(log_message)

def save_plot(user_id):
    face_scores = [score[0] for score in similarity_scores]
    voice_scores = [score[1] for score in similarity_scores]
    
    # Create the histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(face_scores, bins=10, alpha=0.5, label='Face Similarity')
    plt.hist(voice_scores, bins=10, alpha=0.5, label='Voice Similarity')
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Ensure the 'plots' directory exists
    plot_directory = os.path.join(os.getcwd(), 'plots')
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Save the plot in the 'plots' folder with user_id as filename
    plot_path = os.path.join(plot_directory, f'{user_id}.png')
    plt.savefig(plot_path)

    # Provide feedback about plot location
    logging.info(f"Plot saved at {plot_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/store', methods=['GET', 'POST'])
def store():
    if request.method == 'POST':
        name = request.form['name']
        rollno = request.form['rollno']
        
        pincode = request.form['pincode']
        face_data = request.form['face_data']
        voice_data = request.form['voice_data']
        
        user_id = generate_unique_user_id()  # Generate a unique 7-digit user ID

        # Save face and voice data to files
        face_filename = f"data/{user_id}.jpg"
        voice_filename = f"data/{user_id}.wav"
        with open(face_filename, 'wb') as f:
            f.write(base64.b64decode(face_data))
        with open(voice_filename, 'wb') as f:
            f.write(base64.b64decode(voice_data))
        
        encrypted_face_data = encrypt_data(face_filename)
        encrypted_voice_data = encrypt_data(voice_filename)
        
        user_data = {
            'user_id': user_id,
            'name': name,
            'rollno': rollno,
            
            'pincode': pincode,
            'face_data': encrypted_face_data,
            'voice_data': encrypted_voice_data
        }
        
        collection.insert_one(user_data)
        create_log_entry("Registration", user_id, f"User {name} registered with roll number {rollno}")

        return redirect(url_for('welcome', user_id=user_id, name=name, rollno=rollno))
    return render_template('store.html')

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        user_id = request.form['user_id']
        pincode = request.form['pincode']
        face_data = request.form['face_data']
        voice_data = request.form['voice_data']

        user = collection.find_one({'user_id': user_id})

        if user:
            encrypted_face_data = user['face_data']
            encrypted_voice_data = user['voice_data']
            stored_text = user.get('predefined_text', '')  # Retrieve stored predefined text

            face_filename = decrypt_data(encrypted_face_data)
            voice_filename = decrypt_data(encrypted_voice_data)

            similarity_percentage_face = calculate_similarity(face_filename, face_data) * 100

            # **Capture spoken text from the user**
            spoken_text = capture_speech_text()

            similarity_percentage_voice, text_match = calculate_voice_similarity(voice_filename, voice_data, stored_text, spoken_text)
            similarity_percentage_voice *= 100

            similarity_scores.append((similarity_percentage_face, similarity_percentage_voice))
            save_plot(user_id)

            # Define thresholds
            threshold_face = 50.0  # 50% threshold for face
            threshold_voice = 50.0  # 50% threshold for voice

            # Count how many parameters match
            match_count = 0
            if similarity_percentage_face >= threshold_face:
                match_count += 1
                
            if similarity_percentage_voice >= threshold_voice and text_match:  # Ensure text also matches
                match_count += 1

            if verify_pincode(user['pincode'], pincode):
                match_count += 1

            # if user['pincode'] == pincode:
            #     match_count += 1

            # Grant access if at least two parameters match
            if match_count >= 2:
                create_log_entry("Successful Entry", user_id, 
                                 f"User recognized with {similarity_percentage_face:.2f}% face similarity and {similarity_percentage_voice:.2f}% voice similarity")
                return redirect(url_for('welcome', user_id=user_id, name=user['name'], rollno=user['rollno']))
            else:
                message = "Access Denied: At least two parameters must match."
                create_log_entry("Unauthorized Entry Attempt", user_id, message)
                return message
        else:
            message = "User ID not found."
            create_log_entry("Unauthorized Entry Attempt", user_id, message)
            return render_template('recognize.html', error=message)

    return render_template('recognize.html')

def capture_speech_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    try:
        with mic as source:
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Say the predefined phrase!")
            audio = recognizer.listen(source, phrase_time_limit=5)
        
        spoken_text = recognizer.recognize_google(audio).lower()
        return spoken_text
    except Exception as e:
        print(f"Error capturing speech: {e}")
        return ""

@app.route('/welcome')
def welcome():
    user_id = request.args.get('user_id')
    name = request.args.get('name')
    rollno = request.args.get('rollno')
    return render_template('welcome.html', user_id=user_id, name=name, rollno=rollno)

@app.route('/capture_face_data', methods=['GET'])
def capture_face():
    face_img = capture_face_data()
    if face_img:
        return jsonify({'success': True, 'face_data': face_img})
    return jsonify({'success': False})

@app.route('/capture_voice_data', methods=['GET'])
def capture_voice():
    voice_data = capture_voice_data()
    if voice_data:
        with open('data/temp.wav', 'rb') as f:
            voice_data = f.read()
        return jsonify({'success': True, 'voice_data': base64.b64encode(voice_data).decode('utf-8')})
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(debug=True)