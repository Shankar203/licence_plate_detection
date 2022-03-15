import json
from rest_framework.renderers import JSONRenderer
from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import serializers, status
from rest_framework.views import APIView

from .serializers import DetectSerializer
from . import detect


class Detect(APIView):
    serializer_class = DetectSerializer
    def post(self, request):
        image = request.FILES['image']
        results = detect.label_gen(image.read())
        return Response({"results":results},status=status.HTTP_200_OK)
