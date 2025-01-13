from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'), 
    path('main/', views.main_page_view, name='main'),
    path('rate/', views.rate_movies_view, name='rate'),
    path('recommendations/', views.recommendations_view, name='recommendations'),
    path('logout/', views.logout_view, name='logout'),
    path('datasets/', views.datasets_view, name='datasets'),
    path('model/', views.train_model_view, name='model'),
    path("test_model/", views.test_model_view, name="test_model"),  
    path('file-browser/', views.file_browser_view, name='file_browser'),
    path('terminal-output/', views.terminal_output_view, name='terminal_output'),
    path('get-file-content/', views.get_file_content, name='get_file_content'),
    path('save-file-content/', views.save_file_content, name='save_file_content'),


]
