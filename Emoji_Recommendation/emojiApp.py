import pandas as pd 
import numpy as np 
import streamlit as st 

import joblib 
model = joblib.load(open("emojiClassifier.pkl","rb"))

def predict_emotions(sentence):
    results = model.predict([sentence])
    return results[0]

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    st.title("Emoji Recommendation")
    sentence = st.text_area("Type Sentence Here")
    if st.button('Predict Emoji'):
        prediction = predict_emotions(sentence)
        emoji_icon = emotions_emoji_dict[prediction]
        st.write(emoji_icon)

if __name__ == '__main__':
    main()