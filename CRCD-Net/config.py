import os


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret')
    DEBUG = os.getenv('FLASK_ENV', 'development') == 'development'
