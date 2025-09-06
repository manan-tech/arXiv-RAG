import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
import feedparser
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['LANGSMITH_PROJECT'] = "WebRAG"
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"

st.set_page_config(layout="wide")
st.title("🔍 Arxiv RAG with LLMs")

with st.sidebar:
    st.header("⚙️ Settings")
    MODEL_OPTIONS = {
        "Llama 4 Scout": "meta-llama/llama-4-scout:free",
        "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
        "Deepseek R1": "deepseek/deepseek-r1:free",
        "GPT OSS 20B": "openai/gpt-oss-20b:free"
    }
    model_choice = st.selectbox(
        "Select LLM Model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index("Llama 4 Scout")
    )
    num_docs = st.slider("Number of Arxiv documents to load", min_value=1, max_value=10, value=5)

query = st.text_input("Enter your search query (Arxiv title)")

@st.cache_resource
def search_arxiv(title, max_results=5):
    base_url = "https://export.arxiv.org/api/query?"
    query_str = f"search_query=ti:{title}&start=0&max_results={max_results}"
    response = requests.get(base_url + query_str)
    feed = feedparser.parse(response.text)
    return [entry.link.split("/")[-1] for entry in feed.entries]

@st.cache_resource
def create_vectorstore(query, max_docs):
    step_search = st.empty()
    step_split = st.empty()
    step_embedding = st.empty()

    step_search.info("🔹 Searching Arxiv and loading documents...")
    arxiv_ids = search_arxiv(query, max_results=max_docs)
    docs = []
    for arxiv_id in arxiv_ids:
        try:
            loader = ArxivLoader(query=arxiv_id, load_all=True)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["title"] = doc.metadata.get("title", "Unknown Title")
            docs.extend(loaded_docs)
        except:
            continue
    step_search.success("✅ Arxiv documents loaded!")

    if not docs:
        st.warning("No documents found for this query.")
        return None, None, []

    step_split.info("🔹 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    step_split.success("✅ Documents split into chunks!")

    step_embedding.info("🔹 Creating embeddings and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_db = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_db.as_retriever()
    result = vector_db.similarity_search(query, k=max_docs)
    step_embedding.success("✅ Embeddings and FAISS vector store created!")

    return result, retriever, arxiv_ids

if query:
    result, retriever, arxiv_ids = create_vectorstore(query, num_docs)
    if retriever is not None:
        llm_step = st.empty()
        llm_step.info("🔹 Generating answer from LLM...")

        llm = ChatOpenAI(
            model=MODEL_OPTIONS[model_choice],
            openai_api_key=os.environ['OPENAI_API_KEY'],
            openai_api_base="https://openrouter.ai/api/v1"
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question from your knowledge and given context. Answers should be well phrased and easy to understand and long enough:
            <context>
            {context}
            </context>
            <question>
            {input}
            </question>
            """
        )

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
            document_variable_name="context"
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": query})
        llm_step.success("✅ LLM answer generated!")

        st.subheader("📝 LLM Answer")
        st.write(response['answer'])

        st.subheader("📄 Top Relevant Papers")
        for arxiv_id in arxiv_ids:
            try:
                loader = ArxivLoader(query=arxiv_id)
                docs = loader.load()
                if docs:
                    metadata = docs[0].metadata 
                    title = metadata.get("Title", "No Title")
                    summary = metadata.get("Summary", "")  

                    st.markdown(f"**{title}** -- **{arxiv_id}**")
                    with st.expander("Summary"):
                        st.write(summary)
            except Exception as e:
                st.warning(f"Failed to load {arxiv_id}: {e}")
                continue
else:
    st.warning("Please enter a search query.")