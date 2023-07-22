from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate

# initializing the model as Llama 1.0 using 4 bit quantization
llm = LlamaCpp(model_path="./models/llama-7b.ggmlv3.q4_0.bin")
embeddings = LlamaCppEmbeddings(model_path="./models/llama-7b.ggmlv3.q4_0.bin")


template = """I want you to answer the question to the best of your knowledge while keeping in mind the context that has been provided below. 
If you dont know the answer then just say you don't know, do not make up an answer

{context}

Question: {question}
Answer: 
"""
prompt = PromptTemplate.from_template(template=template)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Let me supply the file DBT
loader = TextLoader(file_path="./temp/dbt.txt")
docs = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(docs)

# pushing to chroma vector store
db = Chroma.from_documents(texts, embedding=embeddings)

query1 = "What is a sparse index?"
query2 = "What is a dense index?"
query3 = "Give me an example on a sequential file"


def search(
    db,
    query,
):
    similar_doc = db.similarity_search(query, k=1)
    context = similar_doc[0].page_content
    query_llm = LLMChain(llm=llm, prompt=prompt)
    response = query_llm.run({"context": context, "question": query})
    print(response)


search(db, query=query1)
search(db, query=query2)
search(db, query=query3)
