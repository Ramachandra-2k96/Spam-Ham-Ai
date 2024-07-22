from django.http import JsonResponse
from django.shortcuts import render
import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import os
from django.conf import settings
from collections import Counter

# Load your pre-trained model and vectorizer
with open(os.path.join(settings.BASE_DIR,'spam_classifier.pkl'), 'rb')as model_file:
    model1 = pickle.load(model_file)
with open(os.path.join(settings.BASE_DIR,'spam_classifier_SVC.pkl'), 'rb')as model_file:
    model2 = pickle.load(model_file)
with open(os.path.join(settings.BASE_DIR,'spam_classifier_XGBClassifier.pkl'), 'rb')as model_file:
    model3 = pickle.load(model_file)
  
    
with open(os.path.join(settings.BASE_DIR,'vectorizer.pkl'), 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    
# Create your views here.
def home(request):
	return render(request,'app1/index.html')

def classify_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        text = data.get('text', '')

        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)

        # Transform the text using the loaded vectorizer
        text_vectorized = vectorizer.transform([text])
        
        # Predict using the loaded model
        prediction1 = model1.predict(text_vectorized)[0]
        prediction2 = model2.predict(text_vectorized)[0]        
        prediction3 = model3.predict(text_vectorized)[0]
        predictions=[prediction1,prediction2,prediction3]
        counter = Counter(predictions)
        most_common_prediction = counter.most_common(1)[0][0]
        return JsonResponse({'result': int(most_common_prediction)})

    return JsonResponse({'error': 'Invalid request method'}, status=405)