import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

class Document:
    def __init__(self, page_content):
        self.page_content = page_content
        self.metadata = metadata or {}

def main():
    st.header("Chat with PDF 💬")

    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        
        documents = [Document(page_content=chunk, metadata={"source": pdf.name}) for chunk in chunks]

        store_name = pdf.name[:-4]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}_chunks.pkl"):
            with open(f"{store_name}_chunks.pkl", "rb") as f:
                documents = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            with open(f"{store_name}_chunks.pkl", "wb") as f:
                pickle.dump(documents, f)

        query = st.text_input("Ask questions about your PDF file:")

        if query:
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=documents, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()