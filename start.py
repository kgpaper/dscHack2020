from flask import Flask, request, session, url_for, redirect, render_template, g, flash
from UI_API import similar_words, return_PN
app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def init(word=None):
    if request.method == 'POST':
        name = request.form["word"]
        return render_template('res.html', word=name)
    return render_template('index.html')

@app.route('/res')
def res():
    return render_template('res.html')

if __name__ == '__main__':
    app.run()
