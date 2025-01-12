from langchain_community.vectorstores import SingleStoreDB
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings
import os

def create_qa_chain(cursor, connection) -> RetrievalQAWithSourcesChain:
    # Initialize LangChain components
    llm = ChatAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model_name='claude-3-5-sonnet-20241022',
        temperature=0.8
    )

    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

    # Initialize SingleStore vector store
    vector_store = SingleStoreDB(
        embedding=embeddings,
        host=os.environ["SINGLESTORE_HOST"],
        port=os.environ["SINGLESTORE_PORT"],
        user=os.environ["SINGLESTORE_USER"],
        password=os.environ["SINGLESTORE_PASSWORD"],
        database=os.environ["SINGLESTORE_DATABASE"],
        table_name="document_embeddings",
        vector_field="embedding",  # Field containing the embeddings
        content_field="text_representation",  # Field containing the text
        metadata_field="metadata" # Field containing the metadata
    )

    # Create retrieval chain
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        verbose=True,
        return_source_documents=True
    )

    # Close the cursor and connection
    cursor.close()
    connection.close()

    return qa