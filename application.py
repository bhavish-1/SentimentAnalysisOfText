from flask import Flask, render_template, request
from main import main

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')


@application.route('/sentiment_prediction', methods=['POST'])
def sentiment_prediction():
    if request.method == 'POST':
        text = request.form['text'] 
        prediction = main(text)

        if prediction==1:
            return render_template('index.html', prediction='Positive')
        else :
            return render_template('index.html', prediction='Negative')


if __name__ == '__main__':
    application.run(debug=True)