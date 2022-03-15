from rest_framework import serializers


class DetectSerializer(serializers.Serializer):
    image = serializers.ImageField()
