from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def home(name='home'):
    return render_template('index.html', name=name)

@app.route('/eda')
def eda(name='eda'):
    return render_template('florida_profile.html', name=name)

if __name__ == '__main__':
    app.run('0.0.0.0', 8085)