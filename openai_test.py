import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from PyPDF2 import PdfReader

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./openAIDemo"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

reader = PdfReader("book/Coaching_for_High_Performance.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
rag.insert(text)

# Perform hybrid search
print(rag.query("What were the results of the 2006 Fortune study on coaching?", param=QueryParam(mode="hybrid")))