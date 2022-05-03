from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
import cv2
from PIL import Image
import base64
import numpy as np
import io
from . import generator

model = generator.create_model()

def data_uri_to_cv2_img(uri):
    content_type = uri.split(',')[0]
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img , content_type

# Create your views here.
@csrf_exempt
def hello(request):
    img = data_uri_to_cv2_img(img_global)
    cv2.imshow('sample image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return HttpResponse("Hello world")

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        image = json.loads(request.body)["image"]
        img,content_type = data_uri_to_cv2_img(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = generator.predict(model,img)
        return JsonResponse(img,safe = False)
