import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Données d'exemple
data = [
    ['Where are your headquarters located?', 'location'],
    ['My office, Mistral IA, is located in the city of Paris', 'location'],
    ['Throw my cellphone in the water', 'random'],
    ['Network Access Control?', 'networking'],
    ['Address', 'location']
]
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

'''

# Texte de recherche
search_text = 'where is your office?'
search_vector = encoder.encode(search_text)
_vector = np.array([search_vector])

# Normalisation du vecteur de recherche
faiss.normalize_L2(_vector)

# Effectuer la recherche dans l'index
k = index.ntotal
distances, ann = index.search(_vector, k=k)

# Créer un DataFrame des résultats
results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})

# Fusionner avec les données d'origine pour obtenir les réponses correspondantes
merged = pd.merge(results, df, left_on='ann', right_index=True)

# Trier par distances (du plus proche au plus éloigné)
merged_sorted = merged.sort_values(by='distances')

# Afficher les résultats dans le terminal
for idx, row in merged_sorted.iterrows():
    print(f"Text: {row['text']} | Category: {row['category']} | Distance: {row['distances']:.4f}")


#labels  = df['category']
#category = labels[ann[0][0]]
'''