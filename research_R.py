#from ChromaDB_database import collection, encoder




# Fonction pour rechercher les documents les plus proches d'une requête
def search_similar_documents(collection, encoder, query, top_k=5):
    # Encodage de la requête en vecteur
    query_vector = encoder.encode([query])[0]
    
    # Recherche dans la base de données ChromaDB
    results = collection.query(
        query_embeddings=[query_vector.tolist()],  # Vecteur de la requête
        n_results=top_k  # Nombre de résultats à renvoyer
    )
    return results

def print_research_results(results):
    # Affichage des résultats
    for result in results['documents']:
        print(f"Document similaire trouvé : {result}")
    

