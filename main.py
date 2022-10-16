from flask import Flask, render_template, request
import json
import urllib


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    output = json.loads(output)
    response = urllib.request.urlopen(output)
    with open('image.jpg', 'wb') as f:
        f.write(response.file.read())
