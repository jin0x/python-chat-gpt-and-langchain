from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

import pysqlite3 as sqlite3
import sys
sys.modules['sqlite3'] = sqlite3
print(sqlite3.sqlite_version)


load_dotenv()

embeddings = OpenAIEmbeddings()

emb = embeddings.embed_query("hi there")

print(emb)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0,

)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# db = Chroma.from_documents(
#     docs,
#     embedding=embeddings,
#     persist_directory="emb"
# )

# results = db.similarity_search_with_score("What is an interesting fact about the English language?")

# for result in results:
#     print("\n")
#     print(result[1])
#     print(result[0].page_content)

# for doc in docs:
#     print("\n")
#     print(doc.page_content)
