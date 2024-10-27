import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess
import os
import PyPDF2

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Configuration de FAISS et initialisation de la base de données
data = []

# Dossier contenant les fichiers PDF
pdf_folder = 'data/pdfs'

# Catégories associées aux fichiers PDF (dans l'ordre des fichiers dans le dossier)
categories = ['literature', 'history']

# Charger les PDF et extraire le texte
pdf_files = os.listdir(pdf_folder)
for idx, pdf_file in enumerate(pdf_files):
    pdf_path = os.path.join(pdf_folder, pdf_file)
    pdf_text = extract_text_from_pdf(pdf_path)
    category = categories[idx] if idx < len(categories) else 'unknown'
    data.append([pdf_text, category])

# Créer un DataFrame avec les données extraites
df = pd.DataFrame(data, columns=['text', 'category'])

# Encodage des textes en vecteurs
text = df['text']
encoder = SentenceTransformer("paraphrase-mpnet-base-v2")
vectors = encoder.encode(text)

# Définir les dimensions des vecteurs
vector_dimension = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dimension)

# Normalisation des vecteurs et ajout à l'index FAISS
faiss.normalize_L2(vectors)
index.add(vectors)

# Capturer le prompt de l'utilisateur
prompt = input("Enter your question: ")

# Recherche dans FAISS avant de passer au LLM
search_vector = encoder.encode([prompt])
faiss.normalize_L2(search_vector)

k = index.ntotal
distances, ann = index.search(search_vector, k=k)

# Créer un DataFrame des résultats
results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
merged = pd.merge(results, df, left_on='ann', right_index=True)
merged_sorted = merged.sort_values(by='distances')

# Récupérer les documents pertinents
context_documents = "\n".join(merged_sorted['text'].tolist())

# Passer les documents pertinents à Ollama avec le prompt
full_prompt = f"{context_documents}\n\nUser prompt: {prompt}"

# Utiliser Ollama avec le modèle Mistral pour générer une réponse
result = subprocess.run(
    ["ollama", "run", "mistral", full_prompt],
    capture_output=True,
    text=True
)

# Afficher la réponse générée
print("LLM Response:")
print(result.stdout)
