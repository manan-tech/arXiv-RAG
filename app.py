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

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    MODEL_OPTIONS = {
        "Llama 4 Scout": "meta-llama/llama-4-scout:free",
        "Llama 4 Maverick": "meta-llama/llama-4-maverick:free",
        "Deepseek R1": "deepseek/deepseek-r1:free",
        "GPT OSS 20B": "openai/gpt-oss-20b:free",
    }
    model_choice = st.selectbox(
        "Select LLM Model",
        list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index("Llama 4 Scout"),
    )
    num_docs = st.slider("Number of Arxiv documents to load", min_value=1, max_value=10, value=5)

query = st.text_input("Enter your search query (Arxiv title)")

# Search function with HTTPS
@st.cache_data
def search_arxiv(title, max_results=5):
    base_url = "https://export.arxiv.org/api/query?"
    query_str = f"search_query=ti:{title}&start=0&max_results={max_results}"
    response = requests.get(base_url + query_str, timeout=30)
    feed = feedparser.parse(response.text)
    return [entry.link.split("/")[-1] for entry in feed.entries]

# Hybrid doc loader (ArxivLoader -> fallback feedparser)
@st.cache_data
def load_papers(arxiv_ids):
    papers = []
    for arxiv_id in arxiv_ids:
        try:
            loader = ArxivLoader(query=arxiv_id, load_all=True)
            docs = loader.load()
            if docs:
                papers.append({
                    "title": docs[0].metadata.get("Title", docs[0].metadata.get("title", "No Title")),
                    "summary": docs[0].metadata.get("Summary", docs[0].metadata.get("summary", docs[0].page_content))
                })
                continue
        except Exception:
            pass

        # fallback: use feedparser directly
        try:
            url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            resp = requests.get(url, timeout=30)
            feed = feedparser.parse(resp.text)
            if feed.entries:
                papers.append({
                    "title": feed.entries[0].title,
                    "summary": feed.entries[0].summary
                })
        except Exception:
            continue
    return papers

# Vectorstore creation
@st.cache_resource
def create_vectorstore(query, max_docs):
    step_search = st.empty()
    step_split = st.empty()
    step_embedding = st.empty()

    step_search.info("🔹 Searching Arxiv and loading documents...")
    arxiv_ids = search_arxiv(query, max_results=max_docs)
    papers = load_papers(arxiv_ids)

    if not papers:
        st.warning("No documents found for this query.")
        return None, None, []

    docs = []
    for paper in papers:
        from langchain.schema import Document
        docs.append(Document(page_content=paper["summary"], metadata={"title": paper["title"]}))

    step_search.success("✅ Arxiv documents loaded!")

    step_split.info("🔹 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    step_split.success("✅ Documents split into chunks!")

    step_embedding.info("🔹 Creating embeddings and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_db.as_retriever()
    result = vector_db.similarity_search(query, k=max_docs)
    step_embedding.success("✅ Embeddings and FAISS vector store created!")

    return result, retriever, papers

# Run app
if query:
    result, retriever, papers = create_vectorstore(query, num_docs)
    if retriever is not None:
        llm_step = st.empty()
        llm_step.info("🔹 Generating answer from LLM...")

        llm = ChatOpenAI(
            model=MODEL_OPTIONS[model_choice],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_api_base="https://openrouter.ai/api/v1",
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question from your knowledge and given context. 
            Answers should be well phrased and easy to understand and long enough:
            <context>
            {context}
            </context>
            <question>
            {input}
            </question>
            """
        )

        document_chain = create_stuff_documents_chain(
            llm=llm, prompt=prompt, document_variable_name="context"
        )

        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": query})

        llm_step.success("✅ LLM answer generated!")

        st.subheader("📝 LLM Answer")
        st.write(response["answer"])

        st.subheader("📄 Top Relevant Papers")
        for paper in papers:
            st.markdown(f"**{paper['title']}**")
            with st.expander("Summary"):
                st.write(paper["summary"])