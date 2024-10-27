from django.shortcuts import render
from django.http import JsonResponse
import subprocess

def query_llm(request):
    if request.method == "POST":
        query = request.POST.get('query')
        
        # Commande pour appeler Ollama avec le modèle Mistral
        result = subprocess.run(
            ["ollama", "run", "mistral", "-p", query],
            capture_output=True,
            text=True
        )
        
        # Extraire la première phrase de la réponse
        response = result.stdout
        first_sentence = response.split('.')[0] + '.'
        
        # Retourner la réponse et la première phrase dans le contexte
        return render(request, 'query_result.html', {
            'response': response,
            'first_sentence': first_sentence
        })
    return render(request, 'query_form.html')
