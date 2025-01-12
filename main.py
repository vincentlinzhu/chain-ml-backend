from utils.embeddings import embed_docs
from utils.singlestore_write import write_to_singlestore
from utils.inference import create_qa_chain, query_vector_store
from utils.data_loading import load_docs
from utils.transform_docs import tableglom
from tests.equake_taiwan_test import run_tests
from sycamore.utils.pdf_utils import show_pages
from sycamore.data import Element
from sycamore.transforms.partition import ArynPartitioner

# API Keys
import os
# from google.colab import userdata
from dotenv import load_dotenv
# Helper functions and objects
from sycamore.llms.openai import OpenAIModels, OpenAI
from sycamore.data import Element, Document
from PIL import Image, ImageOps
import sycamore
from sycamore.context import ExecMode
from sycamore.materialize import AutoMaterialize

load_dotenv()

def main():
    #LOAD DOCS
    docs = load_docs("s3://current-events-supply-chain")

    # PARTITIONING
    partitioned_doc = docs.partition(ArynPartitioner(extract_table_structure=True, extract_images=True, use_ocr=True))

    # TRANSFORMING
    doc_transformed = partitioned_doc.map(tableglom)
    # doc_transformed.show()

    # EMBEDDING
    embedded_docs = embed_docs(doc_transformed) # Still needs .take_all()

    # WRITE TO SINGLESTORE
    cursor, connection = write_to_singlestore(embedded_docs)
    
    # INFERENCE
    qa = create_qa_chain(cursor, connection)
    
    query = "What are likely outcomes to the US supply chain after new tariff policies?"
    result, qa_response = query_vector_store(query, qa)
    print(result)
    print(qa_response)

if __name__ == "__main__":
    main()
    