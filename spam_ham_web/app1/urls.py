from django.urls import path
from app1.views import home,classify_text
urlpatterns = [
	path('',home),
 	path('classify_text', classify_text, name='classify_text'),
]