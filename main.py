from utils.embeddings import embed_docs
from utils.singlestore_write import write_to_singlestore
from utils.inference import create_qa_chain
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
    docs = load_docs("./data/")

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
    
    run_tests(qa)

if __name__ == "__main__":
    main()
    