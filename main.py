from flask import Flask, flash, redirect, render_template,request, url_for
from flask_wtf import FlaskForm
from wtforms import TextAreaField,SubmitField
from wtforms.validators import DataRequired
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np

embed = hub.load('https://tfhub.dev/google/Wiki-words-250/2')
model = load_model('model.h5')
app = Flask(__name__)

app.config['SECRET_KEY'] = '73cb8543424f49f42263a072b2ae64b1'

class Form(FlaskForm):
    title = TextAreaField('Title',validators=[DataRequired()])
    submit = SubmitField('Predict')


def text_preprocessing(text):
    import string
    from nltk.tokenize import word_tokenize
    
    text = text.lower()
    
    for punc in string.punctuation:
        text = text.replace(punc,'')
        
    text_words = word_tokenize(text)
    
#     filtered_words = [word for word in text_words if word not in stopwords.words('english')]
    
#     ps = PorterStemmer()
    
#     final_text = ' '.join([ps.stem(word) for word in filtered_words])
    
    return ' '.join(text_words)

def embedding(text):
    text_list = text.split(' ')
    embedded_text = embed(text_list)
    return embedded_text

def padding_with_zeros(text,maxlen=20):
    if len(text)<maxlen:
        new_text = np.concatenate([np.zeros([maxlen-len(text),250]),text])
    else:
        new_text = text[:maxlen]
    return np.array([new_text])

def predict(text):
    preprocessed_text = text_preprocessing(text)
    embed_text = embedding(preprocessed_text)
    padded_text = padding_with_zeros(embed_text)
    prediction = (model.predict(padded_text)>0.5).astype(int)
    prediction_txt = "Fake News" if prediction==1 else "True News" 
    return prediction_txt
@app.route('/',methods=['GET','POST'])
# @app.route('/home',methods=['GET','POST'])
def home():
    form = Form()
    print(request.method)
    title_msg = ''
    pred_msg = ''
    
    if form.validate_on_submit():
        title = form.title.data
        prediction = predict(title)
        title_msg = f'Title : {title}'
        pred_msg = f'Prediction: {prediction}'

    form.title.data = ''
    return render_template('home.html',form=form,title_msg=title_msg, pred_msg=pred_msg)

if __name__ == "__main__":
    app.run(debug=True)

# text = input('Enter here :')
# print(predict(text)[0][0])