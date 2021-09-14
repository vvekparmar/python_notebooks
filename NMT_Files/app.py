import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('s2s.h5')
  return model
with st.spinner('Model is being loaded...'):
  model = load_model()

def collectData():
  userData = st.text_input("Enter your data : ")
  return userData

def predictData(userData, model):
  prediction = model.predict(userData)
  return prediction

model = load_model()
userData = collectData()

if st.button('Translate'):
    result = predictData(userData, model)
    st.write('Result :- ',result)