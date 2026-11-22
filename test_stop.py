from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from agent import agent


def test(frage, limit):
    print(f"\nFrage: {frage}\nLimit: {limit}")
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=frage)]},
            config={"recursion_limit": limit},
        )
        for m in result["messages"]:
            print(" -", type(m).__name__, getattr(m, "tool_calls", "") or "")
        print("Antwort:", result["messages"][-1].content)
        return False
    except GraphRecursionError as e:
        print("GraphRecursionError:", e)
        return True


frage = "Suche im Web den Bitcoin-Preis und prüfe ihn mit CoinGecko und einer zweiten Websuche, bis alle Werte übereinstimmen."
fehler = 0
for i in range(5):
    print(f"\n=== Lauf {i + 1} ===")
    if test(frage, 6):
        fehler += 1

print(f"\nErgebnis: {fehler} mal GraphRecursionError, {5 - fehler} mal Antwort (Limit 6, 5 Läufe)")