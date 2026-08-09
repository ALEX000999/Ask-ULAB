import os
import sys
from pathlib import Path
import gradio as gr

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

print("Starting Ask ULAB...")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("ERROR: GROQ_API_KEY not set!")
    sys.exit(1)

file_path = "ULAB_Data.txt"
Chroma_DB_Path = "./ulab_chroma_db"
Embedding_Model = "all-MiniLM-L6-v2"
LLM_Model = "llama-3.3-70b-versatile"
Top_K_Result = 10

def load_documents(file_path):
    docs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line and line.startswith('ULAB_'):
                doc_id, content = line.split(':', 1)
                docs.append(Document(
                    page_content=content.strip(),
                    metadata={'id': doc_id.strip(), 'line': line_num}
                ))
    print(f"Loaded {len(docs)} documents")
    return docs

documents = load_documents(file_path)

def create_vector_store(docs, embed_model, db_path):
    print(f"Loading {embed_model}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Creating DB with {len(docs)} docs...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    print(f"Ready! {db._collection.count()} docs stored")
    return db

vector_store = create_vector_store(documents, Embedding_Model, Chroma_DB_Path)

def build_rag_chain(vector_store, llm_model, api_key, top_k=5):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    standalone_template = """Given the chat history and a follow-up question,
rephrase the follow-up question to be a standalone question.

Chat History: {chat_history}
Follow Up: {question}
Standalone Question:"""
    standalone_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=standalone_template
    )

    answer_template = """You are Ask ULAB, the friendly AI assistant for the University of Liberal Arts Bangladesh (ULAB).

STRICT RULES:
1. Answer ONLY using the context provided below.
2. NEVER guess or add information not in the context.
3. If the answer is NOT in the context, say exactly:
   "I don't have that specific information. Please contact ULAB at admissions@ulab.edu.bd or call 01714-161613."
4. Format answers clearly — use bullet points or numbered lists when listing multiple items.
5. At the end of EVERY answer, add ONE helpful follow-up suggestion like:
   "Would you like to know about [related topic]?"
6. Keep answers friendly, clear, and concise.

Context:
{context}

Question: {question}

Answer:"""
    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=answer_template
    )

    llm = ChatGroq(
        model=llm_model,
        api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
    )
    print(f"LLM ready: {llm_model}")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=standalone_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        return_source_documents=True,
        verbose=False
    )
    print("RAG chain ready!")
    return chain

rag_chain = build_rag_chain(vector_store, LLM_Model, GROQ_API_KEY, Top_K_Result)

gradio_history = []

def chat_with_ulab(message, history):
    global gradio_history
    if not message.strip():
        return "Please enter a valid question."
    try:
        result = rag_chain.invoke({
            "question": message,
            "chat_history": gradio_history
        })
        answer = result.get("answer") or result.get("result") or "No answer returned."
        gradio_history.append((message, answer))
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_with_ulab,
    title="Ask ULAB",
    description="The Chat Bot for University of Liberal Arts Bangladesh",
    examples=[
        "What is the CSE tuition fee?",
        "What scholarship can I get with GPA 5.0?",
        "How many departments does ULAB have?",
        "Where is ULAB permanent campus?",
        "What clubs are available at ULAB?"
    ]
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
