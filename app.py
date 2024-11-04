from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import sklearn
import nltk
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

app = Flask(__name__)

@app.route('/api/procesar', methods=['POST'])
def procesar():
    data = request.json
    # Lógica del proyecto_conciencia_artificial
    resultado = "Resultado del procesamiento con tus librerías"
    return jsonify({'resultado': resultado})

if __name__ == '__main__':
    app.run(debug=True)