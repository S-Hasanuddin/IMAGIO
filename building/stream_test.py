from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from chrome.Chrome import db

def upload_and_rag(message:str) -> str:
    """Handles both document uploads and fusion RAG (semantic + BM25)."""
  
    user_input = message

    prompt_template = PromptTemplate(template="Answer the question based on the context provided.\n if no context is provided or if the context is not relevant to the question, then answer the question genuinely and in detail.\n Context :{context}\n\n question: {question}")

    try:
        # 1) semantic retriever from Chroma
        raw_docs = db.get(include=["documents", "metadatas"])
        # Convert raw docs to LangChain Document objects
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
        ]

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents=documents, k=5)

        # Similarity retriever
        similarity_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )

        # Combine with EnsembleRetriever
        ensemble = EnsembleRetriever(
            retrievers=[similarity_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
        # Retrieve documents
        docs = ensemble.invoke(user_input)
        print(len(docs))
 
        context = "\n\n".join(doc.page_content for doc in docs)

        # 8) Format the prompt with the context and user question
        final_prompt = prompt_template.format(context=context, question=user_input)

        # 9) Return the context and prompt
        return final_prompt
    except Exception as e:
        return "Error rag"




from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from chrome.Chrome import db

def ingest_document(file_path: str) -> None:
    """
    Load a document, split it into chunks, and add to Chroma DB.

    Args:
        file_path (str): Path to the input file.
        db (Chroma): Chroma database instance.
    """
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"source": file_path})]
        elif file_path.lower().endswith(".csv"):
            loader = CSVLoader(file_path)
            docs = loader.load()
        elif file_path.lower().endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file_path)
            docs = loader.load()
        else:
            raise ValueError("Unsupported file type. Only .pdf, .txt, .csv, .xlsx are supported.")

        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=25)
        chunks = splitter.split_documents(docs)
        db.add_documents(chunks)
        print(f"✅ Successfully ingested {len(chunks)} chunks from {file_path}")
    except Exception as e:
        print(f"❌ Error ingesting document: {e}")
