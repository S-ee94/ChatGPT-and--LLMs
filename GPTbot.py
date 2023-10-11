# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 07:03:07 2023

@author: seema
"""


import streamlit as st
import random
import time
import openai
from transformers import BioGptTokenizer, BioGptForCausalLM
from transformers import pipeline, set_seed

# Set your OpenAI API key
openai.api_key = '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'

# Load the pretrained healthcare model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
healthcare_model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

st.title("Biomedical Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("How can I help you?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the user input contains healthcare-related keywords
    healthcare_keywords = ["pain", "cancer", "specialist", "patient", "emergency", "treatment", "medical", "health",
                           "clinic", "stress", "muscle", "diet", "symptom"]
    is_healthcare_related = any(keyword in prompt.lower() for keyword in healthcare_keywords)

    if is_healthcare_related:
        # Process user input using the healthcare model
        inputs = tokenizer(prompt, return_tensors='pt')
        healthcare_response = healthcare_model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

        # Generate a response using ChatGPT
        chatgpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Choose an appropriate model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Healthcare Assistant: {tokenizer.decode(healthcare_response[0])}\nUser:"},
            ],
            max_tokens=None,  # Adjust the max_tokens as needed to control the response length
        )

        # Extract and display the ChatGPT response
        chatgpt_response_text = chatgpt_response['choices'][0]['message']['content']
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            # Simulate stream of response with milliseconds delay
            for chunk in chatgpt_response_text.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        # Generate a non-healthcare-related response
        non_healthcare_response = "I am a Biomedical bot, please ask only health/medical related questions."
        with st.chat_message("assistant"):
            st.markdown(non_healthcare_response)
        # Add non-healthcare-related response to chat history
        st.session_state.messages.append({"role": "assistant", "content": non_healthcare_response})























