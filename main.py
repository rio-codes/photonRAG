import os
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm import *
from lightrag.utils import EmbeddingFunc
from PyPDF2 import PdfReader

WORKING_DIR = "./photonDemo"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "L3-8B-Lunar-Stheno.Q4_K_M-HF",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://127.0.0.1:5000/v1",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="sentence-transformers/all-mpnet-base-v2",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://127.0.0.1:5000/v1"
    )

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=embedding_func
    )
)

reader = PdfReader("book/Coaching_for_High_Performance.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
rag.insert(text)

# Perform hybrid search
print(rag.query("What were the results of the 2006 Fortune study on coaching?", param=QueryParam(mode="hybrid")))