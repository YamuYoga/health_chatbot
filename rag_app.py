
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Setting up text splitter
chunkiness = 800
splitter = RecursiveCharacterTextSplitter(chunk_size=chunkiness, chunk_overlap=150)

hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/content/"
vector_db_path = "/content/faiss_index"

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Load Vector Database
vector_db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

# Retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

memory = ConversationBufferMemory(memory_key='chat_history',
                                  return_messages=True,
                                  output_key='answer')  # Set output_key to 'answer'

template = """You are a nice chatbot having a conversation with a human. Answer the question based only on the following context and previous conversation. Keep your answers short and succinct.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:"""

# Initialize default LLM and prompt template
llm = HuggingFaceEndpoint(repo_id=hf_model)
prompt = PromptTemplate(template=template,
                        input_variables=["context", "question"])

if "llm" not in st.session_state:
    st.session_state.llm = HuggingFaceEndpoint(
        repo_id=hf_model,
        temperature=0.01,
        repetition_penalty=1.03,
        top_p=0.95
    )
    st.session_state.prompt = PromptTemplate(template="""You are a helpful chatbot having a conversation with a human. Answer the question based on the following context and previous conversation. Answer like you are a nutritionist.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:""", input_variables=["context", "question"])

    st.session_state.memory = memory
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        st.session_state.llm,
        retriever=retriever,
        memory=st.session_state.memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": st.session_state.prompt}
    )

# Function to update LLM and prompt template
def update_llm_and_prompt():
    global llm, prompt, chain, memory
    temperature = st.session_state.my_temperature
    llm = HuggingFaceEndpoint(
        repo_id=hf_model,
        temperature=temperature,
        repetition_penalty=1.03,
        top_p=0.95
    )
    st.session_state.prompt = PromptTemplate(template="""You are a """+st.session_state.my_selectbox+""" having a conversation with a human. Answer the question based on the following context and previous conversation. Answer like you are a nutritionist.

Previous conversation:
{chat_history}

Context to answer question:
{context}

New human question: {question}
Response:""", input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key='chat_history',
                                      return_messages=True,
                                      output_key='answer')
    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": st.session_state.prompt}
    )
    st.session_state.memory = memory
    st.session_state.chain = chain

##### Streamlit #####
st.title('Whole foods Nutrition Chatbot')

def form_callback():
    update_llm_and_prompt()
    # Reset chat history in session state
    st.session_state.messages = []

col1, col2 = st.columns([1, 3], gap='large')

with col2:
    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Curious minds wanted!"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Begin spinner before answering question so it's there for the duration
        with st.spinner("Going down the rabbithole for answers..."):
            # send question to chain to get answer
            answer = st.session_state.chain(prompt)
            # extract answer from dictionary returned by chain
            response = answer["answer"]
            # Display chatbot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer["answer"])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

with col1:
    st.header('Chatbot Tuning')
    st.slider('Creativity', min_value=0.01, max_value=2.0, value=0.01, key='my_temperature', on_change=form_callback)
    st.write(f'Temperature set to: {st.session_state.my_temperature}')
    st.selectbox('Personality', ['Nutritionist', 'Sassy Teenager', 'Hippie'], key='my_selectbox', on_change=form_callback)
    st.write(f'Personality: {st.session_state.my_selectbox}')
