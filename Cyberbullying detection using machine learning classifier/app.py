from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib as jb


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	data =pd.read_csv('labeled_tweets.csv')
	data['label']=data['label'].map({'Non-offensive':0,'Offensive':1 })
	df_data= data[["full_text","label"]]
	# Features and Labels
	df_x = df_data['full_text']
	df_y = df_data['label']
    # Extract Feature With CountVectorizer
	corpus = df_x
	cv = CountVectorizer()
	X = cv.fit_transform(corpus) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.20, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	NB = MultinomialNB()
	NB.fit(X_train,y_train)
	NB.score(X_test,y_test)
	#Alternative Usage of Saved Model
	#ytb_model = open("Naives_model.pkl","rb")
	#NB = joblib.load(ytb_model)

	if request.method == 'POST':
		comment = request.form['comment']
		data = [comment]
		vect = cv.transform(data).toarray()
		my_prediction = NB.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)