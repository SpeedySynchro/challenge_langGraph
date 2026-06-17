"""Der Graph: LLM-Knoten + Werkzeug-Schleife.

Aufbau:
  1. LLM-Knoten  - liest den State und entscheidet, ob ein Werkzeug noetig ist
  2. Werkzeug-Knoten - fuehrt das bestellte Werkzeug aus
  3. Bedingte Kante  - prueft nach jedem LLM-Aufruf: fertig oder Werkzeug?
  4. Schleife        - Werkzeug-Ergebnis geht zurueck zum LLM
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, END

from tools import alle_werkzeuge

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Das LLM - das Gehirn des Agenten
# ---------------------------------------------------------------------------

llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

llm_mit_werkzeugen = llm.bind_tools(alle_werkzeuge)


# ---------------------------------------------------------------------------
# 2. Die zwei Knoten des Graphen
# ---------------------------------------------------------------------------

def llm_knoten(state: MessagesState) -> dict:
    """Liest den gesamten State (alle bisherigen Nachrichten) und
    erzeugt entweder eine finale Antwort oder einen Tool-Call-Bestellzettel."""
    antwort = llm_mit_werkzeugen.invoke(state["messages"])
    return {"messages": [antwort]}


werkzeug_knoten = ToolNode(tools=alle_werkzeuge)


# ---------------------------------------------------------------------------
# 3. Die bedingte Kante - das Entscheidungskreuz
# ---------------------------------------------------------------------------

def soll_weitermachen(state: MessagesState) -> str:
    """Prueft die letzte Nachricht im State:
    - Hat das LLM einen Bestellzettel (tool_calls) erzeugt? -> Werkzeug ausfuehren
    - Hat das LLM eine normale Antwort gegeben?             -> fertig
    """
    letzte_nachricht = state["messages"][-1]

    if letzte_nachricht.tool_calls:
        return "werkzeuge"
    return END


# ---------------------------------------------------------------------------
# 4. Den Graphen zusammenbauen
# ---------------------------------------------------------------------------


graph_builder = StateGraph(MessagesState)


graph_builder.add_node("llm", llm_knoten)
graph_builder.add_node("werkzeuge", werkzeug_knoten)


graph_builder.set_entry_point("llm")


graph_builder.add_conditional_edges(
    "llm",
    soll_weitermachen,
    {"werkzeuge": "werkzeuge", END: END}
)

graph_builder.add_edge("werkzeuge", "llm")

agent = graph_builder.compile()