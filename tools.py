import os

import requests
from dotenv import load_dotenv
from exa_py import Exa
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ------------------------------------------------------------------------------------------------
# Werkzeug 1: CoinGecko - aktuelle Kryptowaehrungspreise (kein Schluessel noetig)
# ------------------------------------------------------------------------------------------------

@tool
def krypto_preis(coin_id: str) -> str:
    """Liefert den aktuellen Preis einer Kryptowaerung in Euro.

    Nutze die CoinGecko-ID der Waehrung, also z.B. 'bitcoin', 'ethereum' oder 'solana' (nicht das Kuerzel wie BTC oder ETH).
    """

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "eur"}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if coin_id not in data:
        return f"Keine Preisdaten fuer '{coin_id}' gefunden. Bitte die CoinGecko-ID pruefen."

    preis = data[coin_id]["eur"]
    return f"{coin_id} kostet aktuell {preis} EUR."


# ---------------------------------------------------------------------------
# Werkzeug 2: Exa - Websuche nach aktuellen Informationen
# ---------------------------------------------------------------------------

exa = Exa(os.environ.get("EXA_API_KEY"))

@tool
def exa_suche(query: str) -> str:
    """Durchsucht das Web nach aktuellen Informationen, Nachrichten und Fakten.

    Nutze dieses Werkzeug fuer alles, was aktuell ist oder sich haeufig
    aendert - z.B. neueste Nachrichten, aktuelle Ereignisse, frische Fakten.
    """
    ergebnis = exa.search_and_contents(query, num_results=3, text=True)

    treffer = []
    for r in ergebnis.results:
        auszug = (r.text or "")[:500]
        treffer.append(f"Titel: {r.title}\nURL: {r.url}\nAuszug: {auszug}")

    return "\n\n".join(treffer) if treffer else "Keine Web-Ergebnisse gefunden."


# ---------------------------------------------------------------------------
# Werkzeug 3: RAG - Suche in der lokalen Wissensbasis (Chroma)
# ---------------------------------------------------------------------------

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="wissensbasis",
    embedding_function=embeddings,
    persist_directory="chroma_db",
)


@tool
def wissens_suche(query: str) -> str:
    """Durchsucht die interne Wissensbasis (Firmenhandbuch) nach einer Antwort.

    Nutze dies fuer Fragen zu internen Firmeninfos wie Versand, Rueckgabe,
    Garantie oder Kundenservice-Zeiten.
    """
    treffer = vectorstore.similarity_search(query, k=3)
    if not treffer:
        return "Nichts in der Wissensbasis gefunden."
    return "\n\n".join(doc.page_content for doc in treffer)

alle_werkzeuge = [krypto_preis, exa_suche, wissens_suche]


if __name__ == "__main__":
    print(krypto_preis.invoke({"coin_id": "bitcoin"}))

