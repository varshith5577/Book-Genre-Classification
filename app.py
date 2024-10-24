import pickle
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import nltk
from flask import Flask, request, render_template

nltk.download("stopwords")
nltk.download('wordnet')

def cleantext(text):

    text=re.sub(r'[^\w\s]', '', text)
    test =re.sub("[^a-zA-Z]"," ",text)
    text=' '.join(text.split())
    text=text.lower()
    return text

def removestopwords(text):

    stop_words=set(stopwords.words('english'))
    removedstopword= [word for word in text.split(' ') if word not in stop_words]
    return ' '.join(removedstopword)

lemma=WordNetLemmatizer()
def lematizing(sentence):
    stemsentence=""
    for word in sentence.split():
        stem=lemma.lemmatize(word)
        stemsentence +=stem
        stemsentence+= " "
    stemsentence=stemsentence.strip()
    return stemsentence
stemmer=PorterStemmer()
def stemming(sentence):
    stemmed=""
    for word in sentence.split():
        
        stem=stemmer.stem(word)
        stemmed+=stem
        stemmed+=" "
    stemmed= stemmed.strip()
    return stemmed

def test(text,model,tfidf_vectorizer):
    
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)
    
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    
    newmapper = {0: 'Fantasy', 1: 'Science Fiction', 2: 'Crime Fiction',
                 3: 'Historical novel', 4: 'Horror', 5: 'Thriller'}

    return newmapper[predicted[0]]

#loading the model

file= open('book_genre_model.pkl','rb')
model=pickle.load(file)
file.close()

file1= open('tfidf_vectorizer.pkl','rb')
tfidf_vectorizer=pickle.load(file1)
file1.close()


app=Flask(__name__)

@app.route('/',methods=['GET','POST'])

def hello_world():
    if request.method=='POST':
        mydict=request.form

        text=mydict['summary']
        prediction=test(text,model,tfidf_vectorizer)

        return render_template('index2.html',genre=prediction,text=str(text),
                               showresult=True)
    return render_template('index2.html')

if __name__=='__main__':
    app.run(debug='True')