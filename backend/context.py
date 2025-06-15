
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from Chrome import db

def get_and_rag(message:str) -> str:
    """Handles fusion RAG (semantic + BM25)."""
  
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





