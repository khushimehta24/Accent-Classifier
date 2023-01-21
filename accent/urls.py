from django.contrib import admin
from django.urls import path
from django.contrib import admin
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from accent import views

urlpatterns = [
    path('/', views.call_model.as_view()),
]
