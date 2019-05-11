"""mysite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
import mysite.view as view
from django.views.static import serve

urlpatterns = [
    # url('app/index/', view.index), #进入添加页面
    url(r'^$', view.index),  # 网站页面
    url(r'vgg16/', view.detect_image_vgg16),  # 上传函数
    url(r'yolo/', view.detect_image_yolo),
    url(r'yolo_vgg/', view.detect_image_yolo_vgg),
    url(r'image/(?P<path>.*)$', serve, {'document_root': '/home/cups/PycharmProjects/Demo/image'}),
]
