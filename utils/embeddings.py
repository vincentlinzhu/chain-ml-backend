from sycamore.transforms.explode import Explode
from sycamore.transforms.merge_elements import GreedySectionMerger
from sycamore.functions.tokenizer import HuggingFaceTokenizer
from sycamore.transforms.embed import SentenceTransformerEmbedder

def embed_docs(doc_transformed):
    tokenizer = HuggingFaceTokenizer("thenlper/gte-small")
    embedder = SentenceTransformerEmbedder(model_name="thenlper/gte-small", batch_size=100)

    embedded_docs = doc_transformed.explode().embed(embedder)

    embedded_docs.take_all() #converts a doc set --> list of docs, then use a for loop

    return embedded_docs