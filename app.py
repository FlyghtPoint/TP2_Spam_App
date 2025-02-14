from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__)
app.debug = True

# Charger les modèles
cv = pickle.load(open('models/cv.pkl', 'rb'))
clf = pickle.load(open('models/clf.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict_email():
    prediction = None
    if request.method == 'POST':
        email = request.form['email']
        print(email)
        tokenized_email = cv.transform([email])
        prediction = clf.predict(tokenized_email)
        print(prediction)
    return render_template('input.html', prediction=prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email = data.get('email')
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    result = {'prediction': int(prediction[0])}
    return jsonify(result)

if __name__ == '__main__':
    app.run()