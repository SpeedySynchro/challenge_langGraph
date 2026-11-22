# challenge_langGraph

Ein deutschsprachiger KI-Agent, gebaut mit LangGraph. Der Agent beantwortet Fragen und entscheidet selbst, welches von drei Werkzeugen er dafür aufruft: Krypto-Preise in Echtzeit, Websuche oder eine Wissenssuche in eigenen Dokumenten (RAG).

Hochschulprojekt, April bis Mai 2026.

## Was der Agent kann

| Werkzeug | Zweck | Quelle |
|---|---|---|
| `krypto_preis` | Aktueller Preis einer Kryptowährung in EUR | CoinGecko API (ohne API-Key) |
| `exa_suche` | Aktuelle Informationen und Nachrichten aus dem Web | Exa API (3 Treffer, erste 500 Zeichen je Treffer) |
| `wissens_suche` | Antworten aus einem Beispiel-Firmenhandbuch (Versand, Rückgabe, Garantie, Servicezeiten) | Chroma Vektordatenbank |

## Architektur

Der Agent ist eine ReAct-artige Schleife aus zwei Knoten:

```
        +---------+   Werkzeugaufruf?   +-------------+
START ->|   llm   |-------- ja -------->|  werkzeuge  |
        +---------+<--------------------+-------------+
             |
            nein
             |
            END
```

- **Modell:** `gpt-5-mini` über `ChatOpenAI`, `temperature=0`, Werkzeuge per `bind_tools` gebunden.
- **Zustand:** `MessagesState` (die Nachrichtenliste).
- **Knoten `llm`:** ruft das Modell mit allen bisherigen Nachrichten auf.
- **Knoten `werkzeuge`:** ein `ToolNode`, der die vom Modell gewünschten Werkzeuge ausführt.
- **Verzweigung:** Enthält die letzte Nachricht Werkzeugaufrufe (`tool_calls`), geht es zu `werkzeuge` und danach zurück zu `llm`. Sonst endet der Lauf.

Das Sprachmodell führt die Werkzeuge nicht selbst aus. Es entscheidet nur, welches Werkzeug mit welchen Parametern aufgerufen werden soll (Tool Calling). Die Ausführung übernimmt die Anwendung. Die Auswahl hängt allein von der Beschreibung (Docstring) der Werkzeuge ab, es gibt keine zusätzliche Routing-Logik.

## RAG (Wissenssuche)

- **Dokumente:** alle `.md`-Dateien im Ordner `knowledge/` (Beispiel-Firmenhandbuch)
- **Chunking:** `RecursiveCharacterTextSplitter`, 500 Zeichen pro Chunk, 50 Zeichen Überlappung
- **Embeddings:** `text-embedding-3-small` (OpenAI)
- **Vektordatenbank:** Chroma, Collection `wissensbasis`, lokal gespeichert in `chroma_db/`
- **Suche:** Ähnlichkeitssuche, die 3 besten Chunks gehen an das Modell

## Projektstruktur

| Datei | Inhalt |
|---|---|
| `agent.py` | LangGraph-Graph mit Knoten und Verzweigung |
| `tools.py` | Die drei Werkzeuge |
| `ingest.py` | Liest `knowledge/`, zerlegt die Dokumente und baut die Chroma-Datenbank |
| `main.py` | Startet den Agenten mit drei Beispielfragen (`frage_stellen(frage)`) |
| `test_stop.py` | Testet, ob und wann der Agent aufhört (siehe unten) |
| `knowledge/` | Markdown-Dokumente für die Wissenssuche |
| `requirements.txt` | Abhängigkeiten |
| `.env.example` | Vorlage für die API-Keys |

## Installation und Start

```bash
git clone https://github.com/SpeedySynchro/challenge_langGraph.git
cd challenge_langGraph
python -m venv .venv
```

Umgebung aktivieren:

```bash
# Windows (PowerShell): .venv\Scripts\activate
# Windows (Git Bash):   source .venv/Scripts/activate
# macOS/Linux:          source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

API-Keys eintragen: `.env.example` nach `.env` kopieren und diese Werte setzen:

```
OPENAI_API_KEY=...
EXA_API_KEY=...
```

Wissensdatenbank einmalig aufbauen, danach den Agenten starten:

```bash
python ingest.py
python main.py
```

`ingest.py` muss vor dem ersten Start laufen, weil `tools.py` die Chroma-Datenbank beim Import öffnet. Die Datei `.env` wird nicht ins Repository eingecheckt.

## Beispielfragen

Diese Fragen stellt `main.py`:

- "Was kostet Bitcoin gerade?" (nutzt `krypto_preis`)
- "Wie sind eure Versandkosten?" (nutzt `wissens_suche`)
- "Was sind die aktuellen KI-Neuigkeiten?" (nutzt `exa_suche`)

## Test: Hört der Agent auf?

`test_stop.py` stellt fünfmal dieselbe Aufgabe, die sich nicht exakt lösen lässt: Den Bitcoin-Preis per Websuche und CoinGecko vergleichen, "bis alle Werte übereinstimmen". Das Schrittlimit von LangGraph ist dabei auf 6 gesetzt.

```bash
python test_stop.py
```

Ergebnis eines Testlaufs mit `gpt-5-mini` (November 2026, nur 5 Läufe, also kein belastbarer Durchschnitt):

- 2 Läufe endeten mit einer Antwort: Der Agent verglich die Quellen und fragte nach der gewünschten Toleranz.
- 3 Läufe endeten mit `GraphRecursionError`, weil mehr als 6 Schritte nötig waren.

Gleiche Eingabe und gleiche Einstellungen führen also nicht immer zum gleichen Ablauf. `main.py` fängt den Fehler ab und gibt eine verständliche Meldung aus.

## Bekannte Einschränkungen und nächste Schritte

- Die Wissenssuche gibt immer 3 Treffer zurück, auch wenn keiner zur Frage passt. Ein Schwellenwert für die Ähnlichkeit fehlt noch, damit der Agent auch antworten kann, dass dazu nichts in den Dokumenten steht.
- Kein Reranking der Treffer.
- Kein System-Prompt: Verhalten, Werkzeugwahl und Anredeform (du oder Sie) hängen von den Werkzeugbeschreibungen und dem Modell ab.
- Keine automatisierten Tests, nur Demo-Fragen in `main.py` und das Verhaltens-Skript `test_stop.py`.
