from flask import render_template
from matplotlib.pyplot import title
from app import app, APP_ROOT

@app.route('/')
def home():
  return render_template("index.html", title='Home',)

@app.route('/anylink')
def anylink():
  pass