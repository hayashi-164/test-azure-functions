import azure.functions as func
import logging
import os

from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import CSVLoader, JSONLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_openai import AzureOpenAIEmbeddings

app = func.FunctionApp()

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://endpoint-2.openai.azure.com/"

embed_model = AzureOpenAIEmbeddings(azure_deployment="embedding_model_2")

@app.blob_trigger(arg_name="myblob", path="uploads",
                               connection="testair20240306_STORAGE") 
def rag_index(myblob: func.InputStream):
    # Blob Storage の /uploads にファイルがアップロードされた際に実行される
    # ファイルを元にRAG Indexを作成、作成したIndexフォルダをBlob Storageにアップロードする
    logging.info(f"Python blob trigger function processed blob"
                f"Name: {myblob.name}"
                f"Blob Size: {myblob.length} bytes")

    chunk_size = 100
    chunk_overlap = 50

    connect_str = os.getenv("testair20240306_STORAGE")
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    file_name = myblob.name.replace("uploads/", "")
    
    # azure functionsストレージへの書き込み制限のため/tmpに一時保存する
    tmp_path = "/tmp"
    tmp_file_name = os.path.join(tmp_path, file_name)
    # ファイルの一時保存
    with open(tmp_file_name, "wb") as f:
        f.write(myblob.read())
    if ".csv" in file_name:
        loader = CSVLoader(file_path=tmp_file_name, encoding="utf-8")
    if ".html" in file_name:
        loader = UnstructuredHTMLLoader(file_path=tmp_file_name, encoding="utf-8")
    if ".json" in file_name:
        loader = JSONLoader(file_path=tmp_file_name, encoding="utf-8")
    if ".txt" in file_name:
        loader = TextLoader(file_path=tmp_file_name, encoding="utf-8")
    if ".pdf" in file_name:
        loader = PyPDFLoader(file_path=tmp_file_name, encoding="utf-8")
    else:
        loader = TextLoader(file_path=tmp_file_name, encoding="utf-8")
    raw_docs = loader.load()
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_spliter.split_documents(docs)
    faiss_db = FAISS.from_documents(documents=docs, embedding=embed_model)
    output_path = os.path.join(tmp_path, "vectorstore")
    faiss_db.save_local(output_path)

    # アップロードするコンテナの生成
    # コンテナに使える命名規則のため
    output_container_name = file_name.replace(".", "-") + "-vectorstore"
    blob_service_client.create_container(output_container_name)
    container_client = blob_service_client.get_container_client(output_container_name)

    # アップロード
    files = os.listdir(output_path)
    for file in files:
        with open(file=os.path.join(output_path, file), mode="rb") as data:
            blob_client = container_client.upload_blob(name=file, data=data, overwrite=True)