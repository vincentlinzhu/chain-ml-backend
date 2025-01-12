# from sycamore.connectors.pinecone import PineconeWriter, PineconeWriterClientParams, PineconeWriterTargetParams
# from pinecone import ServerlessSpec
# from pinecone.grpc import PineconeGRPC as Pinecone

from sqlalchemy import create_engine, text
from langchain_community.vectorstores import SingleStoreDB
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings
import os

def write_to_singlestore(embedded_docs) -> None:
    # Connect to SingleStore
    connection_string = f"mysql+pymysql://{os.environ['SINGLESTORE_USER']}:{os.environ['SINGLESTORE_PASSWORD']}@{os.environ['SINGLESTORE_HOST']}:3306/{os.environ['SINGLESTORE_DATABASE']}"
    engine = create_engine(connection_string)

    # Drop existing table if it exists
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS document_embeddings"))
        
        # Create table with vector column
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                id VARCHAR(255) PRIMARY KEY,
                embedding BLOB,
                text_representation TEXT,
                source VARCHAR(255),
                file_path VARCHAR(255),
                vector VECTOR(384)
            )
        """))
        conn.commit()

    # Prepare records for insertion
    records = []
    for d in embedded_docs.take_all():
        if 'embedding' in d:
            records.append({
                'id': d.doc_id,
                'embedding': d.pop('embedding', []),
                'text_representation': d.text_representation,
                'source': 'default',
                'file_path': d['path']
            })

    # Insert records in batches
    batch_size = 20
    with engine.connect() as conn:
        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]
            values = []
            for record in batch:
                embedding_str = ','.join(map(str, record['embedding']))
                values.append(f"""(
                    '{record['id']}',
                    '{embedding_str}',
                    '{record['text_representation']}',
                    '{record['source']}',
                    '{record['file_path']}'
                )""")
            
            insert_query = f"""
                INSERT INTO document_embeddings (id, vector, text_representation, source, file_path)
                VALUES {','.join(values)}
            """
            conn.execute(text(insert_query))
            conn.commit()

    # Initialize LangChain components
    llm = ChatAnthropic(
        anthropic_key=os.environ.get("anthropic_key"),
        model_name='claude-3-5-sonnet-20241022',
        temperature=0.8
    )

    embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    
    # Initialize SingleStore vector store
    vector_store = SingleStoreDB(
        engine,
        embedding_function=embeddings,
        table_name="document_embeddings",
        content_field="text_representation",
        metadata_fields=["source", "file_path"]
    )

    # Create retrieval chain
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        verbose=True,
        return_source_documents=True
    )

    return qa