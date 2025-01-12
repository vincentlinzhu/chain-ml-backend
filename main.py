from utils.embeddings import embed_docs
from utils.singlestore import write_to_singlestore
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

def main():
    pass

if __name__ == "__main__":
    main()
    