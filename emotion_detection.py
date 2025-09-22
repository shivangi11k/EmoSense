!pip install scikit-learn


from sklearn import metrics

#Emotion Detection of Text (Emotion classificatio of text)
 #Text classification
 # Sentiment analysis

import pandas as pd
import numpy as np

!pip install sklearn

import matplotlib.pyplot as plt
import seaborn as sns

!pip install neattext

import neattext.functions as nfx


df = pd.read_csv("emotion_dataset.csv", error_bad_lines=False)


df.head()

df.shape


df.dtypes

df.isnull().sum()

df["Emotion"].value_counts()

df["Emotion"].value_counts().plot(kind = 'bar')

sns.countplot(x = 'Emotion', data = df)



from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "Positive"
    elif sentiment < 0:
        result = "Negative"
    else:
        result = "Neutral"
    return result

get_sentiment('I love coding')

df['Sentiment'] = df['Text'].apply(get_sentiment)


df.head()

df.groupby(['Emotion', 'Sentiment']).size()

df.groupby(['Emotion', 'Sentiment']).size().plot(kind = 'bar')


sns.catplot

sns.catplot(x = 'Emotion', hue = 'Sentiment', data = df,kind = 'count', aspect = 1.5)

dir(nfx)

df['Clean_Text'] = df['Text'].apply(nfx.remove_stopwords)

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_userhandles)

df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_punctuations)

df[['Text', 'Clean_Text']]



from collections import Counter

def extract_keywords(text, num = 50):
    tokens = [tok for tok in text.split()]
    most_common_tokens = Counter(tokens).most_common(num)
    return dict(most_common_tokens)

emotion_list = df['Emotion'].unique().tolist()

emotion_list

joy_list = df[df['Emotion'] == 'joy']['Clean_Text'].tolist()

joy_docx = ' '.join(joy_list)

joy_docx



keyword_joy = extract_keywords(joy_docx)

keyword_joy

def plot_most_common_words(mydict, emotion_name):
    df_01 = pd.DataFrame(mydict.items(),columns = ['token','count'])
    plt.figure(figsize = (20,10))
    plt.title("Plot of {} Most common Keywords".format(emotion_name))
    sns.barplot(x = 'token', y = 'count', data = df_01)
    plt.xticks(rotation = 45)
    plt.show()

plot_most_common_words(keyword_joy, "joy")

surprise_list = df[df['Emotion'] == 'surprise']['Clean_Text'].tolist()
surprise_docx = ' '.join(surprise_list)
keyword_surprise = extract_keywords(surprise_docx)

plot_most_common_words(keyword_surprise, "Surprise")

from wordcloud import WordCloud

def plot_wordcloud(docx):
    mywordcloud = WordCloud().generate(docx)
    plt.figure(figsize = (20, 10))
    plt.imshow(mywordcloud, interpolation = 'bilinear')
    #plt.axis('off')
    plt.show()

plot_wordcloud(joy_docx)

plot_wordcloud(surprise_docx)






from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


from sklearn.model_selection import train_test_split

Xfeatures = df['Clean_Text']
ylabels = df['Emotion']

Xfeatures

cv = CountVectorizer()
cv.fit_transform(Xfeatures)

# Assuming you have already fit and transformed your data using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)

# Get the feature names
cv.get_feature_names_out()

# Now you can access the feature



X.toarray()

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size = 0.2, random_state = 42)





nv_model =MultinomialNB()
nv_model.fit(X_train,y_train)

MultinomialNB()

nv_model.score(X_test, y_test)

y_pred_for_nv = nv_model.predict(X_test)

y_pred_for_nv

sample_text = ["I love coding so much"]

vect = cv.transform(sample_text).toarray()

nv_model.predict(vect)

nv_model.predict_proba(vect)

nv_model.classes_

np.max(nv_model.predict_proba(vect))

def predict_emotion(sample_text,model):
    myvect = cv.transform(sample_text).toarray()
    prediction = model.predict(myvect)
    pred_proba = model.predict_proba(myvect)
    pred_percentage_for_all = dict(zip(model.classes_, pred_proba[0]))
    print("Prediction : {}, Prediction Score : {}".format(prediction[0],np.max(pred_proba)))
    #print(prediction[0])
    return pred_percentage_for_all


predict_emotion(sample_text, nv_model)

predict_emotion(["He hates running all day"], nv_model)



print(classification_report(y_test, y_pred_for_nv))

confusion_matrix(y_test, y_pred_for_nv)

import joblib

model_file = open("EmotionClassifier_nv_model_3_sept.py", "wb")
joblib.dump(nv_model, model_file)
model_file.close()





lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
