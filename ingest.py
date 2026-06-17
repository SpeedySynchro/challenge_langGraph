"""Phase A von RAG: Dokumente in die Chroma-Datenbank laden.

Ablauf:
  1. Dateien aus knowledge/ einlesen
  2. in Haeppchen (Chunks) zerlegen
  3. Embeddings erzeugen (OpenAI)
  4. in der lokalen Chroma-Datenbank (chroma_db/) speichern

Einmal ausfuehren mit:  python ingest.py
"""

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---------------------------------------------------------------------------
# Schritt 1: Dokumente einlesen
# ---------------------------------------------------------------------------

loader = DirectoryLoader(
    path="knowledge",
    glob="**/*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
dokumente = loader.load()
print(f"Schritt 1: {len(dokumente)} Dokument(e) eingelesen.")

# ---------------------------------------------------------------------------
# Schritt 2: In Haeppchen (Chunks) zerlegen
# ---------------------------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(dokumente)
print(f"Schritt 2: {len(chunks)} Haeppchen (Chunks) erzeugt.")

# ---------------------------------------------------------------------------
# Schritt 3 + 4: Embeddings erzeugen und in Chroma speichern
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="wissensbasis",
    persist_directory="chroma_db",
)

print(f"Schritt 3+4: {len(chunks)} Chunks eingebettet und gespeichert.")
print("Fertig! Die Wissensbasis liegt jetzt in chroma_db/")
print("Du kannst den Agenten jetzt starten: python main.py")