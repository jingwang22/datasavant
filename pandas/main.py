import streamlit as st
import pandas as pd
import os
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from PIL import Image
from tempfile import NamedTemporaryFile

def main():
    st.set_page_config(
        page_title="Query My Dataset",
        page_icon=":smiley:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Load company logo and display it
    logo = Image.open("company_logo.png")
    st.image(logo, width=200)

    st.title("Data Savant")
    st.markdown("Welcome to the Data Savant! After you have uploaded your file, please enter your question below and we'll provide you with an answer based on the information in your uploaded CSV file or the default dataset.")

    # Sidebar with instructions
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Enter your OpenAI API key.
    2. Upload your own dataset (optional).
    3. Enter your question in the text input field.
    4. Click the "Ask" button to get an answer based on the CSV data.
    5. The answer will appear below the input field.
    """)

    # API key input
    api_key = st.sidebar.text_input("Enter your OpenAI API key:", type="password")

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file:", type=['csv'])

    # Use uploaded file or default dataset
    if uploaded_file is not None:
        # Save the uploaded file temporarily to be passed to the create_csv_agent function
        temp_file = NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        dataset_path = temp_file.name
    else:
        dataset_path = 'train.csv'

    # Create CSV agent
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        agent = create_csv_agent(OpenAI(temperature=0), dataset_path, verbose=True)
        del os.environ["OPENAI_API_KEY"]
    else:
        st.error("Please enter a valid OpenAI API key.")
        return

    # Get user input
    question = st.text_input("Question:")

    # Run agent on user input
    if st.button("Ask"):
        with st.spinner("Searching for the answer..."):
            answer = agent.run(question)
        st.markdown(f"**Answer:** {answer}")

    # Clean up the temporary file if it was created
    if uploaded_file is not None:
        os.unlink(temp_file.name)

if __name__ == "__main__":
    main()