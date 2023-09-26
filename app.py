import os
import streamlit as st
from html_elements import *
from backend_logic import *
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings


st.set_page_config(page_title="PDF Analyzer",page_icon=':books:')


def main():
    
    st.markdown(foot, unsafe_allow_html=True)
    
    # Add custom CSS for footer
    st.markdown(footer_fix, unsafe_allow_html=True,)
    #st.sidebar.image("img/logo1.png")


    #st.write(inline_margine, unsafe_allow_html=True,)
    st.title("PDF Analyzer")

    
    st.sidebar.title("Menu")
    
    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings", "HuggingFace Embeddings(slower)"])

    
    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", '''"SUPPORT VECTOR MACHINES"'''])

    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if 'openai_api_key' not in st.session_state:
        openai_api_key = st.text_input(
            'Please enter your OpenAI API key or [get one here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
        if openai_api_key:
            st.session_state.openai_api_key = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            #warning_text = 'Please enter your OpenAI API key. Get yours from here: [link](https://platform.openai.com/account/api-keys)'
            #warning_html = f'<span>{warning_text}</span>'
            #st.markdown(warning_html, unsafe_allow_html=True)
            return
    else:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=[
                                      "pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
            if 'eval_set' in st.session_state:
                del st.session_state['eval_set']

        # Load and process the uploaded PDF or TXT files.
        loaded_text = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits = split_texts(loaded_text, chunk_size=1000,
                             overlap=0, split_method=splitter_type)

        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")

        # Embed using OpenAI embeddings
            # Embed using OpenAI embeddings or HuggingFace embeddings
        if embedding_option == "OpenAI Embeddings":
            embeddings = OpenAIEmbeddings()
        elif embedding_option == "HuggingFace Embeddings(slower)":
            # Replace "bert-base-uncased" with the desired HuggingFace model
            embeddings = HuggingFaceEmbeddings()

        retriever = create_retriever(embeddings, splits, retriever_type)


        # Initialize the RetrievalQA chain with streaming output
        callback_handler = StreamingStdOutCallbackHandler()
        callback_manager = CallbackManager([callback_handler])

        chat_openai = ChatOpenAI(
            streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
        qa = RetrievalQA.from_chain_type(llm=chat_openai, retriever=retriever, chain_type="stuff", verbose=True)

        # Check if there are no generated question-answer pairs in the session state
        if 'eval_set' not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 10  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(
                loaded_text, num_eval_questions, 3000)

       # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px; color = "black">{qa_pair['question']}</p>
                    <p style="font-size: 12px; color = "tomato">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # <h4 style="font-size: 14px;">Question {i + 1}:</h4>
            # <h4 style="font-size: 14px;">Answer {i + 1}:</h4>
        st.write("Ready to answer questions.")

        # Question and answering
        user_question = st.text_input("Enter your question:")
        if user_question:
            answer = qa.run(user_question)
            st.write("Answer:", answer)


if __name__ == "__main__":
    main()
