import streamlit as st
from transformers import pipeline

def main():
  st.title("Hugging face model demo")

  input_text=st.text_input("Enter your text", " ")

  model=pipeline("sentiment-analysis",model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",revision="main")

  if st.button("Analyse"):
    result=model(input_text)
    st.write("Prediction:", result[0]['label'],"|score:", result[0]['score'])
    
if __name__=="__main__":
  main()
