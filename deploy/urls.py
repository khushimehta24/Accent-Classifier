from django.contrib import admin
from django.urls import path
from django.contrib import admin
from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from accent import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('model/', views.call_model.as_view()),
    path('modal/', views.call_modal.as_view())
]
