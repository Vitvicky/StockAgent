# saved in colab: https://colab.research.google.com/drive/1kweVyRBo0QdnhWj8VfP0NwiS_jG397ha#scrollTo=wI7rG-sU6Tty

from pypdf import PdfReader
from langchain.document_loaders import UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import spacy
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from openai import OpenAI


# openai_api_key = "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"

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0]) + 1:],
            n_chars)
        
        
def content_loading(path):
    loader = None
    document = None
    if path.lower().endswith(('.docx')):
        loader = Docx2txtLoader(path)
    elif path.lower().endswith(('.pdf')):
        reader = PdfReader(path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        # Filter the empty strings
        fixed_texts = [text for text in pdf_texts if text]
        return_texts = word_wrap(fixed_texts)
    elif path.lower().endswith(('.xlsx', '.xlsm')):
        loader = UnstructuredExcelLoader(path)
    else:
        print("Unknown file type, skipping: ", path)
    if loader is not None:
        document = return_texts
    print('Loaded file: ', path)
    print(document)
    # strip operation
    # pdf_texts = [p.extract_text().strip() for p in document.pages]
    # # Filter the empty strings
    # pdf_texts = [text for text in pdf_texts if text]
    return document


def chunks_to_vector_db(document):
  # chunking process
  chunk_size = 1000
  chunk_overlap_percentage = 15
  text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=round(chunk_size * chunk_overlap_percentage / 100),
        length_function=len,
        is_separator_regex=False,
    )
  token_split_texts = text_splitter.split_documents(document)
  print(f"\nTotal chunks: {len(token_split_texts)}")

  embedding_function = SentenceTransformerEmbeddingFunction()
  chroma_client = chromadb.Client()
  chroma_collection = chroma_client.create_collection("NVDA_annual_report_2024", embedding_function=embedding_function)

  ids = [str(i) for i in range(len(token_split_texts))]

  chroma_collection.add(ids=ids, documents=token_split_texts)
  print(chroma_collection.count())

  return chroma_collection

def retreival_and_generation(chroma_collection,query):
  # retreival
  # query = "What was the total revenue?"

  results = chroma_collection.query(query_texts=[query], n_results=5)
  retrieved_documents = results['documents'][0]

  for document in retrieved_documents:
    print(word_wrap(document))
    print('\n')

  # using searched docs to generate response
  os.environ['OPENAI_API_KEY'] = "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"
  _ = load_dotenv(find_dotenv()) # read local .env file
  openai.api_key = os.environ['OPENAI_API_KEY']
  openai_client = OpenAI()
  gen_model = "gpt-3.5-turbo"
  information = "\n\n".join(retrieved_documents)
  messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]

  response = openai_client.chat.completions.create(
        model=gen_model,
        messages=messages,
    )
  
  content = response.choices[0].message.content
  return content


# main function
input_pdf = "/content/NVIDIAAn.pdf"
document = content_loading(input_pdf)
chroma_collection = chunks_to_vector_db(document)
query = "What was the total revenue for NVIDIA in 2024 Q1?"
response = retreival_and_generation(chroma_collection=chroma_collection, query=query)

print(word_wrap(response))# saved in colab: https://colab.research.google.com/drive/1kweVyRBo0QdnhWj8VfP0NwiS_jG397ha#scrollTo=wI7rG-sU6Tty

from pypdf import PdfReader
from langchain.document_loaders import UnstructuredExcelLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import spacy
import re
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import openai
from openai import OpenAI


# openai_api_key = "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"

def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0]) + 1:],
            n_chars)
        
        
def content_loading(path):
    loader = None
    document = None
    if path.lower().endswith(('.docx')):
        loader = Docx2txtLoader(path)
    elif path.lower().endswith(('.pdf')):
        reader = PdfReader(path)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        # Filter the empty strings
        fixed_texts = [text for text in pdf_texts if text]
        return_texts = word_wrap(fixed_texts)
    elif path.lower().endswith(('.xlsx', '.xlsm')):
        loader = UnstructuredExcelLoader(path)
    else:
        print("Unknown file type, skipping: ", path)
    if loader is not None:
        document = return_texts
    print('Loaded file: ', path)
    print(document)
    # strip operation
    # pdf_texts = [p.extract_text().strip() for p in document.pages]
    # # Filter the empty strings
    # pdf_texts = [text for text in pdf_texts if text]
    return document


def chunks_to_vector_db(document):
  # chunking process
  chunk_size = 1000
  chunk_overlap_percentage = 15
  text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=round(chunk_size * chunk_overlap_percentage / 100),
        length_function=len,
        is_separator_regex=False,
    )
  token_split_texts = text_splitter.split_documents(document)
  print(f"\nTotal chunks: {len(token_split_texts)}")

  embedding_function = SentenceTransformerEmbeddingFunction()
  chroma_client = chromadb.Client()
  chroma_collection = chroma_client.create_collection("NVDA_annual_report_2024", embedding_function=embedding_function)

  ids = [str(i) for i in range(len(token_split_texts))]

  chroma_collection.add(ids=ids, documents=token_split_texts)
  print(chroma_collection.count())

  return chroma_collection

def retreival_and_generation(chroma_collection,query):
  # retreival
  # query = "What was the total revenue?"

  results = chroma_collection.query(query_texts=[query], n_results=5)
  retrieved_documents = results['documents'][0]

  for document in retrieved_documents:
    print(word_wrap(document))
    print('\n')

  # using searched docs to generate response
  os.environ['OPENAI_API_KEY'] = "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"
  _ = load_dotenv(find_dotenv()) # read local .env file
  openai.api_key = os.environ['OPENAI_API_KEY']
  openai_client = OpenAI()
  gen_model = "gpt-3.5-turbo"
  information = "\n\n".join(retrieved_documents)
  messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]

  response = openai_client.chat.completions.create(
        model=gen_model,
        messages=messages,
    )
  
  content = response.choices[0].message.content
  return content


# main function
input_pdf = "/content/NVIDIAAn.pdf"
document = content_loading(input_pdf)
chroma_collection = chunks_to_vector_db(document)
query = "What was the total revenue for NVIDIA in 2024 Q1?"
response = retreival_and_generation(chroma_collection=chroma_collection, query=query)

print(word_wrap(response))