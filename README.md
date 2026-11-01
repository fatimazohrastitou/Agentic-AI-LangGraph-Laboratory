# Practical Lab — Building AI Agents with LangGraph
## Advanced Artificial Intelligence / Generative AI — PhD Students

**Level:** Intermediate  
**Format:** Jupyter Notebook  
**Main technologies:** Python, LangChain, LangGraph, OpenRouter API  
**Optional extension:** Wikipedia API

---

## 1. Project overview

This practical lab develops an intelligent, stateful assistant using LangGraph. It progresses from a minimal state-based workflow to an LLM-powered chatbot, a multi-node workflow, conditional routing, tool calling, conversational memory, and a university-assistant application.

The final assistant is designed to handle several kinds of student questions:

- **COURSES:** course content, syllabus, schedules, exams, grading, prerequisites and study strategies.
- **PROGRAMMING:** programming explanations and code examples.
- **MATHEMATICS:** mathematical questions and calculations.
- **ADMINISTRATION:** enrollment, fees, certificates, procedures and deadlines.
- **GENERAL:** other questions, including requests for current date/time or encyclopedia information.

The assistant uses specialized agent nodes selected by a classifier. Agents can call tools when appropriate. The graph keeps conversation state using a LangGraph checkpointer and a conversation `thread_id`.

### Main objectives

By completing the lab, the student should be able to:

1. Explain the difference between a conventional LLM chain and an agentic workflow.
2. Understand LangGraph concepts: state, nodes, edges, conditional edges and graph execution.
3. Build a simple conversational agent and add memory/state.
4. Create multi-node workflows and route questions conditionally.
5. Integrate tools into an LLM agent.
6. Build and test a small domain-specific assistant.
7. Inspect graph execution and evaluate the assistant with test questions.

---

## 2. Repository contents

```text
LangGraph_Lab/
├── langgraph_lab.ipynb       # Main practical notebook
├── source_code.py            # Consolidated Python implementation
├── README.md                 # This documentation
├── architecture.mmd          # Mermaid architecture diagram
├── short_report.md           # Conceptual questions and explanations
└── screenshots/
    └── README.md             # Screenshot capture checklist
```

The notebook is the primary learning and execution artifact. The Python source file is a consolidated version of the implementation; it does not replace the notebook's explanations, exercise instructions or execution outputs.

---

## 3. Environment and prerequisites

### Software

- Python 3.11 or a compatible recent Python version.
- Jupyter Notebook or JupyterLab.
- Internet access for OpenRouter and Wikipedia.
- An OpenRouter account and API key.

### Python packages

The notebook uses LangGraph, LangChain and provider integrations. Install the dependencies in a notebook cell or terminal:

```bash
python -m pip install -U langgraph langchain langchain-openai langchain-community requests wikipedia ipykernel
```

If running from Jupyter, the equivalent command is:

```python
%pip install -U langgraph langchain langchain-openai langchain-community requests wikipedia
```

After installing or upgrading packages, restart the Jupyter kernel if imports fail, then execute the notebook from the beginning.

### Check installed versions

```python
from importlib.metadata import version

print("LangGraph:", version("langgraph"))
print("LangChain:", version("langchain"))
```

---

## 4. LLM provider configuration — OpenRouter

This version of the notebook connects to OpenRouter using LangChain's OpenAI-compatible chat model integration.

### Create an API key

1. Create or sign in to an OpenRouter account: https://openrouter.ai/
2. Open the API keys page: https://openrouter.ai/settings/keys
3. Create a key and copy it securely.
4. Do not paste the key into source code, screenshots, README files or Git commits.

### Load the key securely

```python
import os
from getpass import getpass

os.environ.pop("OPENAI_API_KEY", None)

if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = getpass(
        "Enter your OpenRouter API key: "
    )
```

If the key was already set as an environment variable, the prompt can be skipped.

### Initialize the chat model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)
```

### Test the connection

```python
response = llm.invoke("Say hello in one short sentence.")
print(response.content)
```

**Model and quota note:** `openrouter/free` is a free-model routing option. The actual model selected, availability, context size, rate limits and tool-calling support can vary. A `429` response means a rate or quota limit has been reached. It does not, by itself, mean the LangGraph implementation is incorrect. If a selected model does not support tool calling, select an available OpenRouter model that advertises tool/function calling support. Free access is not unlimited or guaranteed.

---

## 5. Lab structure and implementation details

### Part I — Environment setup

The environment is prepared by installing LangGraph and LangChain and checking their installed versions. The notebook also asks the student to explain the roles of LangChain, LangGraph and an LLM.

- **LangChain** provides model, prompt, message and tool integrations.
- **LangGraph** provides graph-based orchestration and stateful execution.
- **LLM** means large language model; it interprets input and generates responses, and compatible models can request tool calls.

### Part II — First LangGraph workflow: state

The first exercise defines a typed state:

```python
class AgentState(TypedDict):
    message: str
    response: str
```

A `process_message` node reads `message`, creates a response and returns a partial state update. A `StateGraph` is built with:

1. A `process` node.
2. An edge from `START` to `process`.
3. An edge from `process` to `END`.
4. A compiled graph invoked with an initial state.

This demonstrates that state is the shared data structure passed between nodes.

**Core concepts:**

- **State:** data available to graph nodes.
- **Node:** a function or runnable that reads state and returns updates.
- **Edge:** a transition between nodes.
- **START:** graph entry point.
- **END:** graph termination point.

### Part III — Building an LLM agent

The notebook creates a chatbot node that calls `llm.invoke(...)`. The returned model content is written into the graph state.

The graph contains one chatbot node between `START` and `END`. This exercise demonstrates the difference between a simple deterministic state-processing node and a node that calls an LLM.

For LangChain chat models, the usual text output is accessed with:

```python
response.content
```

The exact content representation can vary by integration, so code should not assume that it is always a list of dictionaries.

### Part IV — Multi-node AI workflow

This exercise implements a two-stage workflow:

```text
User question → Question analyzer → Answer generator → Final answer
```

The analyzer identifies the topic, the user's intention and the information needed. The answer generator receives both the original question and the analysis, then creates the final response.

This separates analysis from response generation and illustrates sequential node execution.

### Part V and Part VI — Conditional routing

The router classifies a question into one of three initial categories:

- `mathematics`
- `programming`
- `artificial_intelligence`

Specialized nodes handle each category:

- `mathematics_agent`
- `programming_agent`
- `ai_agent`

The routing function examines the `category` field and selects the next node. Conditional edges connect the classifier to the corresponding branch. Each specialized node then connects to `END`.

The notebook's later university-assistant capstone expands the categories to COURSES, PROGRAMMING, MATHEMATICS, ADMINISTRATION and GENERAL.

### Part VII — Agent with tools

The lab introduces tools callable by the LLM. A tool is a Python function exposed with a schema and description, commonly using LangChain's `@tool` decorator.

#### Calculator

The notebook discusses a simple `eval()` calculator as an unsafe demonstration. The integrated implementation uses a restricted AST evaluator instead. Only explicitly allowed arithmetic operations are evaluated; arbitrary Python expressions are not executed.

Supported arithmetic operators include addition, subtraction, multiplication, division, exponentiation, modulo, floor division and unary plus/minus.

#### Current date/time

The date/time tool returns the machine's current date and time. Its value depends on the computer's local timezone and clock.

#### Tool execution flow

The LLM is connected to the tools with `llm.bind_tools(tools)`. LangGraph's `ToolNode` executes requested tool calls. The tool result is added to the message history and passed back to the agent, which can then formulate a final answer.

### Conversational memory

The capstone uses a LangGraph checkpointer (`MemorySaver`) and a `thread_id` to identify a conversation.

- Calls using the same `thread_id` can continue the same conversation state.
- A different `thread_id` identifies a separate conversation.
- The in-memory checkpointer is temporary; it does not provide durable storage after the Python process or kernel stops.

The notebook includes a same-thread memory test and a message-trace inspection.

### Error handling

The capstone includes handling for empty input and model/API failures. Tool execution is configured to handle tool errors rather than allowing every tool exception to terminate the workflow. Provider errors such as rate limits should be reported as provider/quota problems, not silently interpreted as successful answers.

---

## 6. Final capstone — Intelligent University Assistant

The capstone combines classification, specialized agents, tools, routing and conversational memory.

### State definition

The assistant state contains:

- `messages`: the conversation messages, updated with LangGraph's `add_messages` reducer.
- `category`: the classifier's selected category.

### Classifier

The classifier reads the latest user message and assigns one category:

| Category | Purpose |
|---|---|
| COURSES | Course content, syllabus, schedules, exams and grading |
| PROGRAMMING | Programming questions and code |
| MATHEMATICS | Mathematical reasoning and calculations |
| ADMINISTRATION | Enrollment, fees, certificates and procedures |
| GENERAL | Other questions |

Empty input is handled separately. If the classifier cannot call the model because of an API failure, the workflow should use a fallback/error path rather than pretending classification succeeded.

### Specialized agents

Each category has its own system prompt and agent node. The prompts establish the role and response behavior of each specialized agent. The mathematics agent is instructed to use the calculator tool for arithmetic. The general agent can use date/time and Wikipedia tools when appropriate.

### Routing and tool loop

The graph follows this high-level flow:

1. Receive a user message.
2. Classify the question.
3. Route to the matching specialized agent.
4. Check whether the agent requested a tool.
5. If a tool is requested, execute it using `ToolNode`.
6. Return the tool result to the agent.
7. Continue until the agent returns a final answer.
8. End the graph execution.

### Optional extension — Wikipedia

The optional extension adds Wikipedia search as a tool. The implementation queries the public MediaWiki API over HTTP, searches for an article, obtains its introduction and returns a limited text extract.

The Wikipedia tool requires internet access. Network failures, non-JSON responses and missing article extracts should be handled as tool errors or readable messages. A successful HTTP status alone does not guarantee that the returned article extract is non-empty.

Wikipedia is an external information source; its content should be checked when accuracy is important.

---

## 7. How to run the project

1. Download or clone the project folder.
2. Install the required Python packages.
3. Open a terminal in the project folder.
4. Start Jupyter:

```bash
jupyter lab
```

5. Open `langgraph_lab.ipynb`.
6. Configure the OpenRouter key in the secure input cell.
7. Execute the notebook from top to bottom.
8. If a package installation requires a kernel restart, restart and rerun the cells in order.
9. Run the test questions and inspect the outputs.
10. Save the notebook after execution so the submission includes genuine outputs.

To run the consolidated Python script instead of the notebook, set `OPENROUTER_API_KEY` in the environment and execute:

```bash
python source_code.py
```

The notebook remains the main course artifact because it includes the exercises, explanations and step-by-step outputs.

---

## 8. Demonstration and testing

The capstone should be tested with at least five different user queries. The lab's demonstration examples cover:

| Example query | Expected category or behavior |
|---|---|
| “What topics are usually covered in a machine learning course?” | COURSES |
| “Write a Python function that computes the factorial of n.” | PROGRAMMING |
| “What is 125 * 38 + 17?” | MATHEMATICS; calculator can be used |
| “How do I request an enrollment certificate?” | ADMINISTRATION |
| “What is today's date?” | GENERAL; date/time tool can be used |
| Empty or whitespace-only input | EMPTY / fallback response |
| “Who is Alan Turing? Search Wikipedia.” | GENERAL; Wikipedia tool can be used |

These are test intentions, not guaranteed outputs. Actual results depend on the configured model, quota, tool-calling support, network access and execution environment. Record the results observed in your own run.

### Suggested additional checks

- Test a second question in the same `thread_id` to verify conversation memory.
- Use a different `thread_id` to verify conversation separation.
- Inspect the message trace to confirm whether a tool call was made.
- Test Wikipedia directly before testing it through the LLM agent.
- Test the calculator independently of the LLM.
- Test empty input and provider/API failure handling.

---

## 9. Screenshots required

Capture screenshots from the actual Jupyter session after running the relevant cells. Do not fabricate outputs. Hide API keys and personal information.

Suggested screenshots:

1. Environment setup and installed package versions.
2. First state-based graph and its output.
3. LLM connection test (without showing the API key).
4. Two-stage workflow output.
5. Conditional routing tests.
6. Calculator and date/time tool tests.
7. Conversational memory test.
8. Final assistant demonstration with multiple categories.
9. Wikipedia direct test or tool-call trace.
10. Graph visualization, if generated by the notebook.

Save the screenshots in the `screenshots/` directory.

---

## 10. Design choices

### Graph-based orchestration

LangGraph was chosen because the project needs explicit state, specialized processing nodes, conditional transitions and a tool execution loop. The graph makes the workflow easier to inspect than a single long function.

### Specialized agents

Separating agents by category keeps instructions focused. It also makes it possible to add or modify a category without rewriting the entire assistant.

### Tool integration

The calculator, date/time and Wikipedia tools provide capabilities that are more reliable or current than asking the LLM to generate every answer from memory. Tool calls are routed through a dedicated tool node.

### OpenRouter integration

OpenRouter provides a common API endpoint for accessing supported chat models. The LangChain OpenAI-compatible client allows the provider to be configured without changing the overall `llm.invoke()` pattern. Model-specific tool support and free-tier quotas remain external constraints.

### Conversational state

The message reducer and checkpointer support multi-turn interactions. A `thread_id` keeps separate conversations isolated.

### Safer arithmetic

The restricted AST calculator avoids unrestricted `eval()` and permits only a defined set of arithmetic syntax and operators.

---

## 11. Limitations and responsible use

- Free API models may impose request, token or concurrency limits.
- A provider's free model may change or become temporarily unavailable.
- Tool calling is not supported by every model; choose a model that explicitly supports it.
- Wikipedia depends on network availability and the quality of its returned article content.
- `MemorySaver` is in-memory and is not a production persistence layer.
- The assistant is a learning prototype, not an official university information system. It must not invent institutional procedures, fees or deadlines.
- API keys must be kept private and excluded from source control and screenshots.

---

## 12. Conclusion

This lab demonstrates the construction of a stateful LLM application with LangGraph. It progresses from a simple graph to multi-node workflows, conditional routing, specialized agents, tool calling and conversational memory. The final university assistant combines these concepts in one application and illustrates how external tools can extend an LLM's capabilities.

The notebook, consolidated source code, architecture diagram, screenshots and short report together document the implementation and its observed behavior.
