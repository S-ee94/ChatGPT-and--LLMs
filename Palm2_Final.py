# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:59:40 2023

@author: Admin
"""

import streamlit as st
import time
import google.generativeai as palm
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import pipeline, set_seed

#for fine tuning - on PALM2 -> text-bison-001
#epochs = 20, batch-size = 16, learning rate = 0.02

defaults = {
  'model': 'models/text-bison-001',
  'temperature': 0.7,
  'candidate_count': 1,
  'top_k': 40,
  'top_p': 0.95,
  'max_output_tokens': 1024,
  'stop_sequences': [],
  'safety_settings': [
    {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1},
    {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1},
    {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2},
    {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2},
    {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2},
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2}
  ],
}

# Configure PALM with your API key
palm.configure(api_key='AIzaSyAHUhVqHI5kUYS0RMeAHjD3TBVgywIePKI')

# Initialize the text generator using the Hugging Face model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

# Initialize the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

st.title("Biomedical Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define healthcare keywords (converted to lowercase)
healthcare_keywords = ["pain", "cancer", "specialist", "patient", "emergency", "treatment", "medical", "health",
                       "clinic", "stress", "muscle", "diet", "symptom"]

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Check if the user input contains healthcare keywords (case-insensitive)
    is_healthcare_related = any(keyword in prompt.lower() for keyword in healthcare_keywords)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if is_healthcare_related:
        # Process user input using the healthcare model
        inputs = tokenizer(prompt, return_tensors='pt')
        healthcare_response = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1,
                                             no_repeat_ngram_size=2)
        healthcare_response_text = tokenizer.decode(healthcare_response[0], skip_special_tokens=True)

        # Use the healthcare response as a prompt for text generation
        generated_text = text_generator(healthcare_response_text)[0]["generated_text"].replace("\n", " ")

        # Use the generated text as a prompt for PALM
        palm_response = palm.generate_text(
            **defaults,
            prompt=generated_text,  # Use the generated text as the prompt
                   
        ).result

        # Add the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": palm_response})
    else:
        # Non-healthcare related question, provide a default response
        default_response = "I am a Biomedical bot, please ask only health/medical related questions."
        st.session_state.messages.append({"role": "assistant", "content": default_response})

# Display the entire chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])