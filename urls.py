from django.urls import path
from . import views

urlpatterns = [
    path('query/', views.query_llm, name='query_llm'),
]
