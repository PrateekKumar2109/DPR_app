"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
import pickle
import os
from langchain.llms import OpenAI
#from embedd import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
import os

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Smart_DPR Demo", page_icon=":robot:")
st.header("Assistant Driller Demo")

uploaded_file = st.file_uploader("Upload a document you would like to chat about", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# check if file is uploaded and file does not exist in data folder
def embed_doc(data_input):
    #check data folder is not empty
    if len(os.listdir("data")) > 0:
        loader = UnstructuredFileLoader(data_input)
        #loader = DirectoryLoader('data', glob="**/*.*")
        raw_documents = loader.load()
        print(len(raw_documents))
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 1000,
            chunk_overlap  = 0,
            length_function = len,
        )
        print("111")
        documents = text_splitter.split_documents(raw_documents)


        # Load Data to vectorstore
        embeddings = CohereEmbeddings()
        print("222")
        vectorstore = FAISS.from_documents(documents, embeddings)
        print("333")

    return vectorstore
vectorstore=embed_doc(uploaded_file)
        # Save vectorstore
        # check if vectorstore.pkl exists
with open("vectorstore.pkl", "wb") as f:
            pickle.dump(vectorstore, f)




# open vectorstore.pkl if it exists in current directory
if "vectorstore.pkl" in os.listdir("."):
    with open("vectorstore.pkl", "rb") as f:
        
        vectorstore = pickle.load(f)
        print("Loading vectorstore...")

    chain = get_chain(vectorstore)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()
def get_text():
    
    input_text = placeholder.text_input("The: ", value="",  key="input")
    return input_text


user_input = get_text()



if st.button("Submit Your Query"):
    # check 
    docs = vectorstore.similarity_search(user_input)
    # if checkbox is checked, print docs

    print(len(docs))
    
    output = chain.run(input=user_input, vectorstore = vectorstore, context=docs[:2], chat_history = [], question= user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
        
    

    st.session_state.past.append(user_input)
    # print(st.session_state.past)
    st.session_state.generated.append(output)
    
    print(st.session_state.generated)
    # PART2 ADDED
    # if st.session_state.generation includes "related topics:" remove that from st.session_state.generation and add it to a new list
    
        
    print(st.session_state.generated)
    print(st.session_state.topics)
    print(type(st.session_state.topics))
# PART 2 ADDED


 

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
