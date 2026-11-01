#!/usr/bin/env python
# coding: utf-8

# # Practical Lab — Building AI Agents with LangGraph
# 
# **Course:** Advanced Artificial Intelligence / Generative AI 
# **Attendees** PHD Students.
# **Technology:** Python, LangChain, LangGraph, LLM APIs  
# **Level:** Intermediate | **Format:** Jupyter Notebook
# 
# ## 1. Lab objectives
# 
# At the end of this lab, you should be able to:
# 
# - Explain the difference between a conventional LLM chain and an agentic workflow.
# - Understand the main LangGraph concepts: **State, Nodes, Edges, Conditional edges, Graph execution**.
# - Build a simple conversational agent and add memory/state to it.
# - Implement conditional routing and integrate tools.
# - Build a multi-step research-style agent that can decide which action to perform.
# - Visualize and test the resulting graph, and evaluate its behavior.
# 
# > **Tip:** Write your answers to the conceptual questions in the *"Your answer"* cells. They form part of your final report.

# ## PART I — Environment setup
# ### Exercise 1 — Install LangGraph

# In[2]:


get_ipython().system('pip install -U langgraph langchain langchain-openai langchain-community')


# In[1]:


from importlib.metadata import version

print("LangGraph:", version("langgraph"))
print("LangChain:", version("langchain"))


# ### Question
# What is the role of:
# 1. **LangChain?**
# 2. **LangGraph?**
# 3. **An LLM?**
# 
# **Your answer:**  
# 1. *...*  
# 2. *...*  
# 3. *...*

# ## PART II — First LangGraph workflow
# ### Exercise 2 — Understanding State
# 
# In LangGraph, the **state** represents the information shared between nodes.

# In[2]:


from typing import TypedDict

class AgentState(TypedDict):
    message: str
    response: str


# In[3]:


def process_message(state: AgentState):
    message = state["message"]
    response = f"Agent received: {message}"
    return {"response": response}


# In[4]:


from langgraph.graph import StateGraph, START, END

graph_builder = StateGraph(AgentState)
graph_builder.add_node("process", process_message)
graph_builder.add_edge(START, "process")
graph_builder.add_edge("process", END)

graph = graph_builder.compile()


# In[5]:


result = graph.invoke({
    "message": "Hello LangGraph!",
    "response": ""
})
print(result)


# **Expected result**
# ```python
# {'message': 'Hello LangGraph!', 'response': 'Agent received: Hello LangGraph!'}
# ```
# 
# ### Questions
# 1. What is a node?
# 2. What is an edge?
# 3. What is `START`?
# 4. What is `END`?
# 5. Why is the state passed between nodes?
# 
# **Your answer:**  
# 1. *...*  
# 2. *...*  
# 3. *...*  
# 4. *...*  
# 5. *...*

# ## PART III — Building an LLM Agent
# ### Exercise 3 — Connect an LLM
# 
# **Do not hard-code the API key in the notebook.** Prefer an environment variable configured outside the notebook; otherwise prompt for it securely as below.

# In[24]:


get_ipython().run_line_magic('pip', 'install -U langchain-google-genai')


# In[36]:


import os
from getpass import getpass

os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

os.environ["GOOGLE_API_KEY"] = getpass(
    "Enter your NEW Gemini API key: "
)

print("Gemini API key loaded.")


# In[39]:


get_ipython().run_line_magic('pip', 'install -U langchain-openai')


# In[91]:


import os
from getpass import getpass

os.environ.pop("OPENAI_API_KEY", None)
os.environ["OPENROUTER_API_KEY"] = getpass(
    "Enter your OpenRouter API Key: "
)


# In[92]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openrouter/free",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)


# In[93]:


response = llm.invoke(
    "Say hello in one short sentence."
)

print(response.content)


# In[18]:


print(response.text)


# ### Exercise 4 — Create an LLM node

# In[64]:


class AgentState(TypedDict):
    message: str
    response: str

def chatbot(state: AgentState):
    response = llm.invoke(state["message"])
    return {"response": response.content}

graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()


# In[65]:


result = graph.invoke({
    "message": "Explain artificial intelligence in three sentences.",
    "response": ""
})
print(result["response"])


# In[36]:


print(result["response"][0]["text"])


# ## PART IV — Multi-node AI workflow
# ### Exercise 5 — Build a two-stage agent
# 
# ```
# User question → Question analyzer → Answer generator → Final answer
# ```

# In[13]:


class AgentState(TypedDict):
    question: str
    analysis: str
    answer: str

def analyze_question(state: AgentState):
    prompt = f"""
    Analyze the following question.

    Question:
    {state['question']}

    Identify:
    1. The topic
    2. The user's intention
    3. The information needed
    """
    response = llm.invoke(prompt)
    return {"analysis": response.content}

def generate_answer(state: AgentState):
    prompt = f"""
    Answer the user's question.

    Question:
    {state['question']}

    Analysis:
    {state['analysis']}

    Give a clear and technically correct answer.
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}


# In[62]:


graph_builder = StateGraph(AgentState)

graph_builder.add_node("analyzer", analyze_question)
graph_builder.add_node("answer", generate_answer)

graph_builder.add_edge(START, "analyzer")
graph_builder.add_edge("analyzer", "answer")
graph_builder.add_edge("answer", END)

graph = graph_builder.compile()


# In[63]:


result = graph.invoke({
    "question": "What is reinforcement learning?",
    "analysis": "",
    "answer": ""
})
print(result["answer"])


# In[42]:


print(result["answer"][0]["text"])


# ## PART V — Conditional routing
# 
# This is where LangGraph becomes particularly interesting. The agent decides which branch to follow:
# 
# ```
#                  ┌─── Mathematics
# Question ────────┼─── Programming
#                  └─── General AI
# ```
# 
# ### Exercise 6 — Create a router

# In[15]:


from typing import TypedDict

class AgentState(TypedDict):
    question: str
    category: str
    answer: str

def classify_question(state: AgentState):
    prompt = f"""
    Classify this question into exactly one category:

    - mathematics
    - programming
    - artificial_intelligence

    Question:
    {state['question']}

    Return only the category.
    """

    response = llm.invoke(prompt)
    content = response.content

    # Gemini peut renvoyer une liste au lieu d'une chaîne
    if isinstance(content, list):
        content = "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    category = content.strip().lower()

    return {"category": category}


# ### Exercise 7 — Create specialized nodes

# In[16]:


def mathematics_agent(state):
    response = llm.invoke(
        f"""
        Solve this mathematics question:

        {state['question']}
        """
    )
    return {"answer": response.content}

def programming_agent(state):
    response = llm.invoke(
        f"""
        Answer this programming question.
        Include Python code when appropriate.

        {state['question']}
        """
    )
    return {"answer": response.content}

def ai_agent(state):
    response = llm.invoke(
        f"""
        Answer this artificial intelligence question
        at university level.

        {state['question']}
        """
    )
    return {"answer": response.content}


# ## PART VI — Conditional edges
# 
# The routing function inspects the state and returns the **name of the next branch**.

# In[17]:


def route_question(state):
    category = state["category"]

    if "mathematics" in category:
        return "math"
    if "programming" in category:
        return "programming"
    return "ai"   # default / fallback branch


# In[18]:


graph_builder = StateGraph(AgentState)

graph_builder.add_node("classifier", classify_question)
graph_builder.add_node("math", mathematics_agent)
graph_builder.add_node("programming", programming_agent)
graph_builder.add_node("ai", ai_agent)

graph_builder.add_edge(START, "classifier")

graph_builder.add_conditional_edges(
    "classifier",
    route_question,
    {
        "math": "math",
        "programming": "programming",
        "ai": "ai"
    }
)

graph_builder.add_edge("math", END)
graph_builder.add_edge("programming", END)
graph_builder.add_edge("ai", END)

router_graph = graph_builder.compile()


# In[59]:


questions = [
    "What is the derivative of x²?",
    "How do I create a REST API with Python?",
    "What is reinforcement learning?"
]

for question in questions:
    result = router_graph.invoke({
        "question": question,
        "category": "",
        "answer": ""
    })
    print("=" * 60)
    print("QUESTION:", question)
    print("CATEGORY:", result["category"])
    print("ANSWER:", result["answer"])


# In[24]:


print("QUESTION:", question)
print("CATEGORY:", result["category"])
print("ANSWER:", result["answer"][0]["text"])


# ## PART VII — Agent with tools
# ### Exercise 8 — Create tools
# 
# The original lab uses `eval()`, which is **dangerous** (it can execute arbitrary Python). Below we first show the lab version for comparison, then a **safe** calculator built on `ast` that only allows arithmetic. We use the safe one from here on.

# In[25]:


from langchain_core.tools import tool

# --- Version from the lab (UNSAFE: do not use with untrusted input) ---
@tool
def calculator_unsafe(expression: str) -> str:
    """Calculate a mathematical expression (unsafe eval version)."""
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"

print(calculator_unsafe.invoke("25 * 4 + 10"))   # Expected: 110


# In[26]:


import ast
import operator as op

_ALLOWED_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv,
    ast.USub: op.neg, ast.UAdd: op.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression such as '25 * 4 + 10'. Supports + - * / ** % //."""
    try:
        return str(_safe_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception:
        return "Invalid expression"

print(calculator.invoke("25 * 4 + 10"))        # Expected: 110
print(calculator.invoke("__import__('os')"))   # Expected: Invalid expression


# In[27]:


from datetime import datetime

@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time. Use for any question about today's date or the current time."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M:%S")

print(current_datetime.invoke(""))


# ## PART VIII — Tool-using agent
# 
# ```
#                  ┌─────────────┐
# Question → Agent ├→ Calculator │
#                  └─────────────┘
#                        ↓
#                     Answer
# ```
# 
# The agent must **decide** whether it needs the calculator. Expected behavior for *"What is 125 * 38?"*:
# 
# `Agent → recognizes math problem → calls calculator → receives result → generates answer`
# 
# We bind the tools to the LLM, so it can emit *tool calls*. A conditional edge sends the flow to the `tools` node when a tool call is present, then loops back to the agent.

# In[44]:


from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

tools = [calculator, current_datetime]
llm_with_tools = llm.bind_tools(tools)

def agent(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_builder = StateGraph(MessagesState)
tool_builder.add_node("agent", agent)
tool_builder.add_node("tools", ToolNode(tools))

tool_builder.add_edge(START, "agent")
tool_builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
tool_builder.add_edge("tools", "agent")                       # loop back with the result

tool_graph = tool_builder.compile()


# In[45]:


for q in ["What is 125 * 38?", "What is the capital of Morocco?"]:
    result = tool_graph.invoke({"messages": [("user", q)]})
    print("=" * 60)
    print("Q:", q)
    for m in result["messages"]:
        m.pretty_print()


# **Observe:** for the first question the trace contains an `AIMessage` with a tool call, a `ToolMessage` with the result, then the final answer. For the second question, no tool is called.

# ## PART IX — Memory
# ### Exercise 9 — Conversational state
# 
# The `add_messages` reducer **appends** new messages to the list instead of overwriting it.

# In[46]:


from typing import Annotated
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

chat_builder = StateGraph(ChatState)
chat_builder.add_node("chatbot", chatbot)
chat_builder.add_edge(START, "chatbot")
chat_builder.add_edge("chatbot", END)


# **Without a checkpointer**, each `invoke` starts from scratch. **With a checkpointer** (`MemorySaver`) and a `thread_id`, the conversation persists between calls.

# In[47]:


# 1) No persistence: the agent forgets
no_memory_graph = chat_builder.compile()
no_memory_graph.invoke({"messages": [("user", "My name is Ahmed.")]})
r = no_memory_graph.invoke({"messages": [("user", "What is my name?")]})
print("WITHOUT memory:", r["messages"][-1].content)


# In[48]:


from langgraph.checkpoint.memory import MemorySaver

# 2) With persistence: the agent remembers within the same thread
memory_graph = chat_builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "ahmed-1"}}

r1 = memory_graph.invoke({"messages": [("user", "My name is Ahmed.")]}, config)
print("Agent:", r1["messages"][-1].content)

r2 = memory_graph.invoke({"messages": [("user", "What is my name?")]}, config)
print("Agent:", r2["messages"][-1].content)

# A different thread has its own, separate memory
other = {"configurable": {"thread_id": "someone-else"}}
r3 = memory_graph.invoke({"messages": [("user", "What is my name?")]}, other)
print("Other thread:", r3["messages"][-1].content)


# ## Visualizing and testing graphs
# 
# `get_graph()` returns a drawable representation. If PNG rendering is unavailable (it may need network access), print the Mermaid source and paste it into <https://mermaid.live>.

# In[49]:


from IPython.display import Image, display

def show_graph(g):
    try:
        display(Image(g.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("PNG rendering unavailable:", e)
        print(g.get_graph().draw_mermaid())

show_graph(router_graph)
show_graph(tool_graph)


# ## Evaluating the behavior of an agentic workflow
# 
# A minimal evaluation harness: define test cases with expected categories, run them, and compute accuracy. You can extend this with tool-call checks, latency, and cost.

# In[50]:


import time

test_cases = [
    ("What is the derivative of x²?", "mathematics"),
    ("Solve 2x + 3 = 11", "mathematics"),
    ("How do I reverse a list in Python?", "programming"),
    ("Write a function to check for palindromes.", "programming"),
    ("What is a convolutional neural network?", "artificial_intelligence"),
    ("Explain overfitting.", "artificial_intelligence"),
]

correct, rows = 0, []
for q, expected in test_cases:
    t0 = time.time()
    out = router_graph.invoke({"question": q, "category": "", "answer": ""})
    got = out["category"]
    ok = expected in got
    correct += ok
    rows.append((q, expected, got, ok, round(time.time() - t0, 2)))

import pandas as pd
df = pd.DataFrame(rows, columns=["question", "expected", "got", "correct", "seconds"])
display(df)
print(f"Routing accuracy: {correct}/{len(test_cases)} = {correct/len(test_cases):.0%}")


# **Discussion:** Where did the router fail (if anywhere)? What would make the routing more robust (structured output, few-shot examples, enum validation)?
# 
# **Your answer:** *...*

# ## PART X — Capstone project: Intelligent University Assistant
# 
# ### Requirements covered below
# | # | Requirement | Implementation |
# |---|---|---|
# | 1 | Question classifier | `classify` node → COURSES / PROGRAMMING / MATHEMATICS / ADMINISTRATION / GENERAL |
# | 2 | Specialized agents | one node per category, built with `make_agent` |
# | 3 | Memory | `MemorySaver` checkpointer + `thread_id` |
# | 4 | Tools | `calculator`, `current_datetime` (+ optional extras) |
# | 5 | Conditional routing | `route_after_classifier`, `route_after_agent`, `route_after_tools` |
# | 6 | Error handling | empty input, unknown category, invalid tool calls, API errors |
# 
# ### Architecture
# ```
# User question → Classifier ─┬→ Courses Agent ──────┐
#                             ├→ Programming Agent ──┤
#                             ├→ Mathematics Agent ──┼→ Tool decision ─→ Tools (Calculator / Date-time / ...)
#                             ├→ Administration Agent┤        │                │
#                             ├→ General Agent ──────┘        ▼                │
#                             └→ Fallback (empty/error)   Final response ◄─────┘
# ```
# Each agent decides whether to call a tool; after the tool runs, control returns to the **same** agent, which writes the final response.

# In[51]:


from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

CATEGORIES = ["COURSES", "PROGRAMMING", "MATHEMATICS", "ADMINISTRATION", "GENERAL"]

class AssistantState(TypedDict):
    messages: Annotated[list, add_messages]
    category: str


# ### 1. Classifier (with error handling)

# In[53]:


def classify(state: AssistantState):
    last = state["messages"][-1]
    text = (last.content if hasattr(last, "content") else str(last)).strip()

    # Error handling: empty question
    if not text:
        return {"category": "EMPTY"}

    prompt = f"""
    Classify the student's question into exactly one category:
    COURSES (course content, syllabus, schedules, exams, grading),
    PROGRAMMING, MATHEMATICS,
    ADMINISTRATION (enrollment, fees, certificates, procedures, deadlines),
    GENERAL (anything else).

    Question: {text}

    Return only the category name.
    """
    try:
        raw = llm.invoke(prompt).content.strip().upper()
    except Exception as e:                      # Error handling: API errors
        print("Classifier API error:", e)
        return {"category": "ERROR"}

    # Error handling: unknown/unexpected label -> GENERAL
    category = next((c for c in CATEGORIES if c in raw), "GENERAL")
    return {"category": category}


# ### 2. Specialized agents (factory) + fallback node

# In[54]:


SYSTEM_PROMPTS = {
    "COURSES": "You are a university courses advisor. Explain course content, prerequisites, "
               "schedules and study strategies. If you lack institution-specific data, say so honestly.",
    "PROGRAMMING": "You are a programming tutor. Give clear explanations and correct Python code when appropriate.",
    "MATHEMATICS": "You are a mathematics tutor. Show steps. ALWAYS use the calculator tool for arithmetic.",
    "ADMINISTRATION": "You are a university administration assistant (enrollment, fees, certificates, procedures). "
                      "Never invent official rules or deadlines; recommend contacting the administration office when unsure.",
    "GENERAL": "You are a helpful university assistant. Use the date/time tool for questions about the current date or time.",
}

def make_agent(category: str):
    system = SystemMessage(content=SYSTEM_PROMPTS[category])

    def node(state: AssistantState):
        try:
            response = llm_with_tools.invoke([system] + state["messages"])
            return {"messages": [response]}
        except Exception as e:                  # Error handling: API errors
            return {"messages": [AIMessage(
                content=f"Sorry, I ran into a technical problem while answering ({type(e).__name__}). Please try again."
            )]}
    return node

def fallback(state: AssistantState):
    if state["category"] == "EMPTY":
        msg = "Your question seems to be empty. Please type a question and I'll help."
    else:
        msg = "Sorry, I couldn't process your question right now. Please try again in a moment."
    return {"messages": [AIMessage(content=msg)]}


# ### 3. Routing functions and graph assembly
# 
# `ToolNode(..., handle_tool_errors=True)` converts invalid tool calls / tool exceptions into error messages the agent can recover from, instead of crashing the graph.

# In[55]:


NODE_NAMES = {c: f"{c.lower()}_agent" for c in CATEGORIES}

def route_after_classifier(state: AssistantState):
    return NODE_NAMES.get(state["category"], "fallback")

def route_after_agent(state: AssistantState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

def route_after_tools(state: AssistantState):
    # return to the agent that requested the tool
    return NODE_NAMES.get(state["category"], "general_agent")

builder = StateGraph(AssistantState)
builder.add_node("classifier", classify)
builder.add_node("fallback", fallback)
builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
for c in CATEGORIES:
    builder.add_node(NODE_NAMES[c], make_agent(c))

builder.add_edge(START, "classifier")
builder.add_conditional_edges(
    "classifier", route_after_classifier,
    {**{n: n for n in NODE_NAMES.values()}, "fallback": "fallback"},
)
for c in CATEGORIES:
    builder.add_conditional_edges(NODE_NAMES[c], route_after_agent, {"tools": "tools", END: END})
builder.add_conditional_edges("tools", route_after_tools, {n: n for n in NODE_NAMES.values()})
builder.add_edge("fallback", END)

assistant = builder.compile(checkpointer=MemorySaver())
show_graph(assistant)


# ### 4. Demonstration — at least 5 different user queries
# Includes: all categories, both tools, memory, and error cases (empty input).

# In[56]:


def extract_gemini_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )

    return str(content)


# In[57]:


from typing import TypedDict

class AgentState(TypedDict):
    question: str
    category: str
    answer: str
def classify_question(state: AgentState):

    prompt = f"""
    Classify this question into exactly one category:

    - mathematics
    - programming
    - artificial_intelligence
    - courses
    - administration
    - general

    Question:
    {state['question']}

    Return only the category.
    """

    response = llm.invoke(prompt)

    content = extract_gemini_text(response.content)

    return {"category": content.strip().lower()}


# In[58]:


def ask(question: str, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}

    result = assistant.invoke(
        {"messages": [HumanMessage(content=question)]},
        config
    )

    print("=" * 70)
    print("USER      :", question)
    print("CATEGORY  :", result["category"])
    print("ASSISTANT :", result["messages"][-1].content)

    return result


demo_queries = [
    "What topics are usually covered in a machine learning course?",
    "Write a Python function that computes the factorial of n.",
    "What is 125 * 38 + 17?",
    "How do I request an enrollment certificate?",
    "What is today's date?",
    "   "
]

for q in demo_queries:
    ask(q)


# In[66]:


# Memory test: same thread_id
ask("My name is Ahmed.", thread_id="memory-test")
ask("What is my name?", thread_id="memory-test");


# In[67]:


# Inspect the full message trace (including tool calls) of one run
r = assistant.invoke(
    {"messages": [HumanMessage(content="Compute (15 + 27) * 3 using the calculator.")]},
    {"configurable": {"thread_id": "trace"}},
)
for m in r["messages"]:
    m.pretty_print()


# ### 5. Optional extensions
# Add one or more of: Wikipedia, web search, a database (e.g. SQLite with course data), a PDF reader, or weather. Implement each as a `@tool`, append it to the `tools` list **before** calling `llm.bind_tools(...)`, and rebuild the graph.
# 
# ```python
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
# wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
# ```

# In[68]:


get_ipython().run_line_magic('pip', 'install -U langchain-community wikipedia')


# In[74]:


import requests

url = "https://en.wikipedia.org/w/api.php"

params = {
    "action": "query",
    "format": "json",
    "titles": "Artificial Intelligence",
    "prop": "extracts",
    "exintro": 1,
    "explaintext": 1
}

headers = {
    "User-Agent": "LangChainAcademicProject/1.0 (Python)"
}

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=20
)

print("Status code:", response.status_code)
print("Content type:", response.headers.get("Content-Type"))
print(response.text[:500])


# In[75]:


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=2000,
    lang="en"
)

wiki = WikipediaQueryRun(
    api_wrapper=wiki_wrapper
)

print("Wikipedia tool initialized successfully!")


# In[78]:


import requests
from langchain_core.tools import tool

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return a short article summary."""

    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AcademicLangChainProject/1.0"
    }

    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 1
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()
    results = data.get("query", {}).get("search", [])

    if not results:
        return "No Wikipedia article found."

    page_id = results[0]["pageid"]

    extract_params = {
        "action": "query",
        "format": "json",
        "pageids": page_id,
        "prop": "extracts",
        "explaintext": 1,
        "exchars": 2000
    }

    response = requests.get(
        url,
        params=extract_params,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    pages = response.json()["query"]["pages"]
    extract = pages[str(page_id)].get("extract", "")

    if not extract:
        return "No article summary available."

    return extract


# In[79]:


result = wikipedia_search.invoke({
    "query": "Artificial Intelligence"
})

print(result)


# In[80]:


tools = [
    calculator,
    current_datetime,
    wiki
]

print("Available tools:")

for tool in tools:
    print("-", tool.name)


# In[81]:


llm_with_tools = llm.bind_tools(tools)


# In[82]:


from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

def agent(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


# In[83]:


tool_builder = StateGraph(MessagesState)

tool_builder.add_node("agent", agent)
tool_builder.add_node("tools", ToolNode(tools))

tool_builder.add_edge(START, "agent")

tool_builder.add_conditional_edges(
    "agent",
    tools_condition
)

tool_builder.add_edge("tools", "agent")

tool_graph = tool_builder.compile()

print("Agent graph compiled successfully!")


# In[95]:


import requests
from langchain_core.tools import tool

@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return an article summary."""

    url = "https://en.wikipedia.org/w/api.php"

    headers = {
        "User-Agent": "AcademicLangChainProject/1.0 (contact: student@example.com)"
    }

    try:
        # Search for the article
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("query", {}).get("search", [])

        if not results:
            return "No Wikipedia article found."

        page_id = results[0]["pageid"]

        # Retrieve the article extract
        params = {
            "action": "query",
            "format": "json",
            "pageids": page_id,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1
        }

        response = requests.get(
            url, params=params, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        page = data["query"]["pages"][str(page_id)]
        extract = page.get("extract", "")

        if not extract:
            return "Article found, but no summary is available."

        return extract[:2000]

    except requests.exceptions.RequestException as e:
        return f"Wikipedia connection error: {e}"
    except (ValueError, KeyError) as e:
        return f"Wikipedia response error: {e}"


# In[96]:


result = wikipedia_search.invoke({
    "query": "Alan Turing"
})

print(result)


# In[97]:


questions = [
    "What is 125 * 38?",
    "What is today's date?",
    "Who is Alan Turing? Search Wikipedia."
]

for q in questions:

    print("\n" + "=" * 70)
    print("QUESTION:", q)

    try:
        result = tool_graph.invoke({
            "messages": [("user", q)]
        })

        print("ANSWER:")
        print(result["messages"][-1].content)

    except Exception as e:
        print("ERROR:", str(e))


# In[98]:


# Test 1: Calculator
print("CALCULATOR:")
print(calculator.invoke({"expression": "125 * 38"}))

# Test 2: Date and time
print("\nDATE/TIME:")
print(current_datetime.invoke({}))

# Test 3: Wikipedia
print("\nWIKIPEDIA:")
print(wikipedia_search.invoke({"query": "Alan Turing"}))


# In[99]:


from IPython.display import Image, display

display(
    Image(assistant.get_graph().draw_mermaid_png())
)


# ## Deliverables checklist
# 
# - [ ] `langgraph_lab.ipynb` (this notebook, executed, with outputs)
# - [ ] `README.md` (setup, how to run, design choices)
# - [ ] Source code
# - [ ] Architecture diagram (export from `show_graph(assistant)` or Mermaid)
# - [ ] Screenshots showing execution
# - [ ] Short report answering the conceptual questions

# In[ ]:




