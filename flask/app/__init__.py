from flask import Flask
import os

# app is the variable for flask application
# static will use the static folder
app=Flask(__name__,static_folder='static')
# serves as layer of security
app.config['SECRET_KEY']='sparta'

APP_ROOT=os.path.dirname(os.path.abspath(__file__))

# use for routing pages
from app import routes