import chromadb

import streamlit as st
import gensim
import gensim.models.word2vec as w2v
from nltk.tokenize import word_tokenize
import string
import os
import fitz
import numpy as np

chroma_client = chromadb.PersistentClient(path="chroma.db")


collection = chroma_client.get_collection("Talkpdf")


def convert_pdf_to_txt(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text_file_path = os.path.splitext(pdf_path)[0] + ".txt"
    with open(text_file_path, "w", encoding="utf-8") as text_file:
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            text_file.write(page.get_text())
    pdf_document.close()
    return text_file_path


st.set_page_config(
    page_title="Talk with your PDFs",
    page_icon="📄",
    layout="wide",
)


col1, col2 = st.columns((1,1))

with col1:
    st.title('Ask The Doc')

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.subheader('Enter your query below:')
    query = st.text_input("Your query :")

    if uploaded_file is not None:
        pdf_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

if uploaded_file is not None and st.button('Get Response'):

    txt_file_path = convert_pdf_to_txt(pdf_path)
        
    chunks = query

 
    def process_text(text):
        tokens = word_tokenize(text)

        tokens = [w.lower() for w in tokens if w.isalpha()]

        return tokens

    processed_chunks = [process_text(chunk) for chunk in chunks]


    model = w2v.Word2Vec(sentences = processed_chunks, vector_size=1000, window=5, min_count=1, workers=4)

    model.train(processed_chunks, total_examples=len(processed_chunks), epochs=10)

    chunk_vectors = []

    for chunk in processed_chunks:

        chunk_vectors.append(sum([model.wv[word] for word in chunk]))



    def query_embedding_generator(query, model):

    
        processed_query = process_text(query)

        query_vector = np.zeros(model.vector_size)

        count=0

        for word in processed_query:

            if word in model.wv:
                query_vector += model.wv[word]

                count+=1
            
            if count >0:
                query_vector /= count

        return query_vector.tolist()


    with open( txt_file_path , 'r', encoding="utf-8") as f:

        chunks = f.readlines()



    query_embedding = query_embedding_generator(query, model)

    results = collection.query(query_embedding, n_results =4)


    responses = []

    for i, result_id in enumerate(results['ids'][0]):
        index = int(result_id[3:]) - 1 
        response_chunk = chunks[index].strip() 
        responses.append(f"Response {i+1}: {response_chunk} (distance: {results['distances'][0][i]})")

    response_text = "\n".join(responses)

    query_prompt = f"User: {query}\n\n{response_text}\n\nGPT-3.5, could you please summarize the responses above? Make sure it is relvant to the query asked"

    import openai

    openai.api_key = "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"


    response_message = openai.Completion.create(
        engine = "text-davinci-002",
        prompt = query_prompt,
        temperature=0.9,
        max_tokens=150
    )

    response = response_message.choices[0].text.strip()
    
    with col2:
        st.subheader('Response:')
        st.write(response)


st.markdown(
    """
    <style>
        body {
            color: #fff;
            background-color: #0e1117;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            color: #0e1117;
            background-color: #f79b8e;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
