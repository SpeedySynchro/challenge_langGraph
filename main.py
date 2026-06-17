"""Einstiegspunkt: Frage stellen und Antwort des Agenten ausgeben."""

from langchain_core.messages import HumanMessage
from agent import agent


def frage_stellen(frage: str) -> str:
    """Schickt eine Frage an den Agenten und gibt die finale Antwort zurueck."""
    print(f"\nFrage: {frage}")
    print("-" * 50)

    ergebnis = agent.invoke({
        "messages": [HumanMessage(content=frage)]
    })

    finale_antwort = ergebnis["messages"][-1].content
    print(f"Antwort: {finale_antwort}")
    return finale_antwort


if __name__ == "__main__":

    frage_stellen("Was kostet Bitcoin gerade?")
    frage_stellen("Wie sind eure Versandkosten?")
    frage_stellen("Was sind die aktuellen KI-Neuigkeiten?")