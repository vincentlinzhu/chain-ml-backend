from utils.embeddings import embed_docs
from utils.singlestore_write import write_to_singlestore

# API keys
import os
from dotenv import load_dotenv

from sycamore.llms.openai import OpenAIModels, OpenAI
from sycamore.data import Element, Document
from PIL import Image, ImageOps
import sycamore
from sycamore.context import ExecMode
from sycamore.materialize import AutoMaterialize

import mysql.connector
import json
import numpy as np

load_dotenv()

#Add your variables in the Secrets page in Colab, and set them here
# ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# ARYN_API_KEY = os.getenv('ARYN_API_KEY')
# SINGLE_STORE_API_KEY = os.getenv('SINGLE_STORE_API_KEY')

# DATA

#uris = ["s3://aryn-public/ntsb/0.pdf", "s3://aryn-public/ntsb/1.pdf", "s3://aryn-public/ntsb/10.pdf", "s3://aryn-public/ntsb/101.pdf", "s3://aryn-public/ntsb/103.pdf", "s3://aryn-public/ntsb/104.pdf", "s3://aryn-public/ntsb/11.pdf", "s3://aryn-public/ntsb/12.pdf", "s3://aryn-public/ntsb/16.pdf", "s3://aryn-public/ntsb/18.pdf", "s3://aryn-public/ntsb/20.pdf", "s3://aryn-public/ntsb/21.pdf", "s3://aryn-public/ntsb/22.pdf", "s3://aryn-public/ntsb/23.pdf"
#, "s3://aryn-public/ntsb/25.pdf", "s3://aryn-public/ntsb/26.pdf", "s3://aryn-public/ntsb/27.pdf", "s3://aryn-public/ntsb/28.pdf", "s3://aryn-public/ntsb/3.pdf", "s3://aryn-public/ntsb/31.pdf", "s3://aryn-public/ntsb/32.pdf", "s3://aryn-public/ntsb/34.pdf", "s3://aryn-public/ntsb/35.pdf", "s3://aryn-public/ntsb/36.pdf", "s3://aryn-public/ntsb/37.pdf", "s3://aryn-public/ntsb/38.pdf", "s3://aryn-public/ntsb/39.pdf"
#, "s3://aryn-public/ntsb/40.pdf", "s3://aryn-public/ntsb/41.pdf", "s3://aryn-public/ntsb/42.pdf"]
uris = "s3://aryn-public/ntsb/"

ctx = sycamore.init(exec_mode=sycamore.EXEC_LOCAL)
ctx.rewrite_rules.append(AutoMaterialize(source_mode=sycamore.MATERIALIZE_USE_STORED))
doc = ctx.read.binary(paths=uris, binary_format="pdf")

from sycamore.utils.pdf_utils import show_pages

show_pages(doc, limit=3)

# PARTITIONING

from sycamore.data import Element
from sycamore.transforms.partition import ArynPartitioner

partitioned_doc = doc.partition(ArynPartitioner(extract_table_structure=True, extract_images=True, use_ocr=True))
# Visualize the partitioned document
show_pages(partitioned_doc, limit=3)

# TRANSFORMING THE DATA

def tableglom(doc):
    first_table = list(filter(lambda e: e.type=='table', doc.elements))[0]
    for elt in doc.elements:
        elt['path'] = doc.properties['path']
        if elt is first_table:
            continue
        elt.text_representation = f"Metadata: {first_table.table.to_csv()}\n{elt.text_representation}"

    return doc

doc_transformed = partitioned_doc.map(tableglom)

doc_transformed.show()

# EMBEDDING
embedded_docs = embed_docs(doc_transformed)

# WRITE TO SINGLESTORE
write_to_singlestore(embedded_docs)

# TESTS

# ## Answered in document 101
# qa.invoke({"question": "How old was the pilot who crashed the flight in Old Bridge NJ?"})['answer']
# qa.invoke({"question": "How old was the pilot who crashed the flight in Old Bridge NJ?"})['source_documents']

# ## Answered in document 25
# qa.invoke({"question": "For the crash with the Registration number N747PK, what caused the fire?"})['answer']

# ## Answered in document 27
# qa.invoke({"question": "Where was the digital flight recorder shipped to for the flight that crashed on January 23, 2023?"})['answer']

# ## Answered in document 42
# qa.invoke({"question": "For the Cessna 172M that crashed in Skull Valley, how long had it been since the last FAA medical exam?"})['answer']
