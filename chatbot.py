from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# Define State
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Define Tools
@tool
def get_stock_price(symbol: str) -> float:
    '''Return the current price of a stock given the stock symbol'''
    return {"MSFT": 200.3, "AAPL": 100.4, "AMZN": 150.0, "RIL": 87.6}.get(symbol, 0.0)


@tool
def buy_stocks(symbol: str, quantity: int, total_price: float) -> str:
    '''Buy stocks given the stock symbol and quantity'''
    decision = interrupt(f"Approve buying {quantity} {symbol} stocks for ${total_price:.2f}?")
    if decision == "yes":
        return f"You bought {quantity} shares of {symbol} for a total price of ${total_price:.2f}"
    else:
        return "Buying declined."


tools = [get_stock_price, buy_stocks]


# Initialize Groq (free, good tool support)
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools)


# Chatbot node
def chatbot_node(state: State):
    msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [msg]}


# Build Graph with Memory
memory = MemorySaver()

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile(checkpointer=memory)


# Run the conversation
config = {"configurable": {"thread_id": "buy_thread"}}

# Step 1: User asks price
print("=" * 50)
print("Step 1: Asking for stock price")
print("=" * 50)
state = graph.invoke(
    {"messages": [{"role": "user", "content": "What is the current price of 10 MSFT stocks?"}]},
    config=config
)
print("Bot:", state["messages"][-1].content)

# Step 2: User asks to buy (triggers interrupt)
print("\n" + "=" * 50)
print("Step 2: Requesting to buy stocks")
print("=" * 50)
state = graph.invoke(
    {"messages": [{"role": "user", "content": "Buy 10 MSFT stocks at current price."}]},
    config=config
)

# Check for interrupt
if state.get("__interrupt__"):
    print("\n⚠️  Approval Required:")
    for intr in state["__interrupt__"]:
        print(f"   {intr.value}")
    
    decision = input("\nApprove (yes/no): ")
    
    # Step 3: Resume with decision
    print("\n" + "=" * 50)
    print("Step 3: Processing decision")
    print("=" * 50)
    state = graph.invoke(Command(resume=decision), config=config)
    print("Bot:", state["messages"][-1].content)
else:
    print("Bot:", state["messages"][-1].content)