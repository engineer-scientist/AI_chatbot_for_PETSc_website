# scripts/index_petsc_docs.py
import pathlib, chromadb, hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# Pick the right loaders for your LangChain version
try:  # LangChain ≥ 0.3.20
    from langchain_community.document_loaders import (
        DirectoryLoader,
        BSHTMLLoader,       # Beautiful-Soup HTML loader
        TextLoader,
    )

    def choose_loader(path: str):
        p = pathlib.Path(path)
        if p.suffix.lower() in {".html", ".htm"}:
            return BSHTMLLoader(str(p))            # needs beautifulsoup4 + lxml
        return TextLoader(str(p), autodetect_encoding=True)

except ImportError:  # Older LangChain
    from langchain.document_loaders import (
        DirectoryLoader,
        HTMLLoader,
        TextLoader,
    )

    def choose_loader(path: str):
        p = pathlib.Path(path)
        if p.suffix.lower() in {".html", ".htm"}:
            return HTMLLoader(str(p), "lxml")
        return TextLoader(str(p), autodetect_encoding=True)

# ---------------------------------------------------------------------
DOCS_DIR = pathlib.Path("../docs_raw")            # your crawled PETSc pages
DB_DIR   = pathlib.Path("../chromadb_petsc")      # will hold the Chroma DB

print("▶️  Loading raw documents …")
loader = DirectoryLoader(str(DOCS_DIR), glob="**/*", loader_cls=choose_loader)
docs   = loader.load()
print(f"   {len(docs):,} files loaded")

print("▶️  Splitting into chunks …")
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunks   = splitter.split_documents(docs)
print(f"   {len(chunks):,} chunks")

print("▶️  Computing embeddings (this can take a few minutes) …")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

client     = chromadb.PersistentClient(str(DB_DIR))
collection = client.get_or_create_collection("petsc-docs")

for i, chunk in enumerate(chunks):
    # deterministic but unique ID (source file name + chunk number)
    uid = f"{pathlib.Path(chunk.metadata['source']).name}-{i}"
    collection.add(
        ids        =[uid],
        embeddings =[embedder.encode(chunk.page_content)],
        documents  =[chunk.page_content],
        metadatas  =[chunk.metadata],
    )

print(f"✔️  Stored {collection.count()} chunks in {DB_DIR}")

