import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb




def database_creation():


    # Fonction pour extraire le texte d'un PDF
    def extract_text_from_pdf(pdf_path):
        with fitz.open(pdf_path) as pdf:
            text = ""
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text += page.get_text("text")
        return text




    # Configuration de l'emplacement des PDF et du modèle d'encodage
    pdf_folder = "./data/pdfs"
    encoder = SentenceTransformer("paraphrase-mpnet-base-v2")

    # Initialisation de la base de données ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("my_database")



    # Parcourir tous les PDF du dossier
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Traitement du fichier : {filename}")
            
            # Extraction du texte du PDF
            text = extract_text_from_pdf(pdf_path)
            
            # Encodage du texte en vecteur
            text_vector = encoder.encode([text])[0]
            
            # Stocker le vecteur dans ChromaDB
            collection.add(
                ids=[filename],  # Ajout de l'ID unique pour chaque document
                documents=[filename],
                metadatas={"filename": filename},
                embeddings=[text_vector.tolist()]  # Assurez-vous de convertir le numpy array en liste
            )
            print(f"Fichier {filename} vectorisé et stocké avec succès.")

    tab_return = [collection, encoder]
    return tab_return

if __name__ == "__main__":

    from research_R import search_similar_documents 
    from research_R import print_research_results

    collection, encoder = database_creation()


    query = str(input("Entrez votre requête : "))
    results = search_similar_documents(collection, encoder, query)
    print_research_results(results)


'''
# Récupération et affichage des 10 premiers vecteurs
results = collection.get(limit=10, include=["embeddings"])

print("Les 10 premiers vecteurs de la base :")
for i, embedding in enumerate(results['embeddings']):
    print(f"Vecteur {i + 1}:")
    print(embedding)
    print("-" * 50)  # Séparateur pour une meilleure lisibilité
'''