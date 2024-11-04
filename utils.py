import os
import numpy as np
from PIL import Image
import torch

def load_images(path, image_size=(64, 64)):
    images = []
    labels = []
    for label_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, label_dir)):
            for image_file in os.listdir(os.path.join(path, label_dir)):
                image_path = os.path.join(path, label_dir, image_file)
                image = Image.open(image_path).convert('RGB')
                image = image.resize(image_size)  # Redimensionar
                image = np.array(image)
                images.append(image)
                labels.append(label_dir)
    return np.array(images), np.array(labels)

def load_audio(path):
    audio_data = []
    labels = []
    for label_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, label_dir)):
            for audio_file in os.listdir(os.path.join(path, label_dir)):
                audio_path = os.path.join(path, label_dir, audio_file)
                # Aquí puedes utilizar una librería como librosa para cargar el audio
                # audio = librosa.load(audio_path, sr=None)
                # audio_data.append(audio)
                labels.append(label_dir)
    return audio_data, labels

def load_texts(path):
    texts = []
    labels = []
    for label_dir in os.listdir(path):
        if os.path.isdir(os.path.join(path, label_dir)):
            for text_file in os.listdir(os.path.join(path, label_dir)):
                text_path = os.path.join(path, label_dir, text_file)
                with open(text_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    texts.append(text)
                labels.append(label_dir)
    return texts, labels

def split_data(data, labels, test_size=0.2, random_seed=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(data, labels, test_size=test_size, random_state=random_seed)

def preprocess_image(image, target_size=(64, 64)):
    image = Image.open(image).convert('RGB')
    image = image.resize(target_size)
    image = np.array(image)
    return image

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model