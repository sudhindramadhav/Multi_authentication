#encryption.py

from Crypto.Cipher import AES
import base64
import hashlib

# Ensure the key length is exactly 16, 24, or 32 bytes
SECRET_KEY = hashlib.sha256('my_secret_key'.encode()).digest()[:16]  # 16 bytes

def encrypt_data(data):
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return base64.b64encode(cipher.nonce + tag + ciphertext).decode('utf-8')

def decrypt_data(encrypted_data):
    data = base64.b64decode(encrypted_data)
    nonce = data[:16]
    tag = data[16:32]
    ciphertext = data[32:]
    cipher = AES.new(SECRET_KEY, AES.MODE_EAX, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag).decode('utf-8')