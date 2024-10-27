import streamlit as st
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as pdf:
        text = ""
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
    return text

# Configuration du modèle d'encodage
@st.cache_resource
def load_encoder():
    return SentenceTransformer("paraphrase-mpnet-base-v2")

# Initialisation de la base de données ChromaDB
@st.cache_resource
def init_chromadb():
    client = chromadb.Client()
    return client.create_collection("my_database")

# Fonction principale de l'application Streamlit
def main():
    st.title("Vectorisation de PDF")

    encoder = load_encoder()
    collection = init_chromadb()

    # Crée le dossier 'data/pdfs' s'il n'existe pas
    os.makedirs("data/pdfs", exist_ok=True)

    # Centrer le bouton d'upload
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Choisissez un fichier PDF", type="pdf")

    if uploaded_file is not None:
        # Sauvegarder le fichier uploadé dans le dossier data/pdfs
        file_path = os.path.join("data/pdfs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Afficher un message pendant la vectorisation
        with st.spinner('Vectorisation en cours...'):
            # Extraction du texte du PDF
            text = extract_text_from_pdf(file_path)

            # Encodage du texte en vecteur
            text_vector = encoder.encode([text])[0]

            # Stocker le vecteur dans ChromaDB
            collection.add(
                ids=[uploaded_file.name],
                documents=[uploaded_file.name],
                metadatas={"filename": uploaded_file.name},
                embeddings=[text_vector.tolist()]
            )

        # Afficher un message de succès
        st.success(f"Le fichier {uploaded_file.name} a été sauvegardé dans 'data/pdfs' et vectorisé avec succès.")

        # Attendre 3 secondes avant de revenir à l'état initial
        import time
        time.sleep(3)

if __name__ == "__main__":
    main()
