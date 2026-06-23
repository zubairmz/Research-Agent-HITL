# Research Agent Pro: Human-in-the-Loop (HITL)

## Problem Statement

Research Agent Pro takes an already-functional multi-tool research agent and elevates it from a fully autonomous system to a human-governed agentic AI. The core upgrade is the integration of **Human-in-the-Loop (HITL)** middleware, which introduces approval gates, edit capabilities, and rejection controls directly into the agent's reasoning-action loop.

This mirrors how production AI systems operate in enterprise environments: the agent proposes actions, the human disposes, and the graph orchestrates the handoff transparently. The result is an agent that is both powerful and trustworthy.

---

## HITL Execution Flow

```
1. User sends query              →  Research question enters the agent
2. Agent reasons & selects tool  →  LLM analyzes via ReAct loop, picks best tool
3. INTERRUPT                     →  State saved to SqliteSaver, awaits human
4. Human reviews tool call       →  Tool name, args, and context displayed
5. Decision: approve/edit/reject →  Three-way control over every tool call
6. Command(resume=True)          →  Execution continues from checkpoint
7. Tool executes (or skips)      →  Approved/edited tool runs; rejected is skipped
8. Agent continues or answers    →  LLM processes result, answers or picks next tool
```

---

## Technology Stack

| Component       | Technology                   |
| --------------- | ---------------------------- |
| Framework       | LangChain 1.x                |
| Graph engine    | LangGraph + SqliteSaver      |
| LLM             | Ollama (local) — qwen3.5:9b  |
| Tools           | DuckDuckGo, Arxiv, Wikipedia |
| Interface       | CLI + real-time streaming    |
| Package manager | uv                           |

---

## Tools Available

| Tool                   | Purpose                                   |
| ---------------------- | ----------------------------------------- |
| `web_search`           | DuckDuckGo — current news and web content |
| `wikipedia`            | Encyclopedia-style facts and summaries    |
| `arxiv`                | Academic papers and scientific research   |
| `get_current_datetime` | Returns the current date and time         |

---

## How HITL Works in This Solution

Every time the agent selects a tool, execution **pauses** before the tool runs. The user sees a checkpoint like this:

```
[HITL Checkpoint]
  Tool : wikipedia
  Args : {'query': 'Lahore'}
  Options: (a)pprove  (e)dit  (r)eject
  Decision:
```

The user then has three choices:

### (a) Approve

Tool runs as-is. Graph resumes with `Command(resume=True)`.

### (e) Edit

User is prompted to modify each argument before the tool runs:

```
  Edit args (press Enter to keep current value):
    query [Lahore]: Lahore history and culture
  -> 'wikipedia' will run with updated args.
```

The updated args are saved into the graph state via `agent.update_state()`. The tool then runs with the new values.

### (r) Reject

User provides a reason. A `ToolMessage` is injected into the graph state with the rejection reason, so the LLM is informed and can respond gracefully — the session does **not** exit.

```
  Reason for rejection: Not relevant to my query
  -> 'wikipedia' rejected. Agent will be notified.
```

---

## Key Implementation Details

### interrupt_before=["tools"]

The agent graph is configured to pause before every tool node execution:

```python
agent = create_agent(
    ...
    interrupt_before=["tools"],
)
```

### Rejection — inject ToolMessage

When rejected, a `ToolMessage` is written into the graph state as if the tools node already ran, so the agent node receives the rejection reason and continues:

```python
agent.update_state(
    config,
    {"messages": [ToolMessage(
        content=f"Tool '{tool_name}' was rejected by user. Reason: {reason}",
        tool_call_id=tc["id"]
    )]},
    as_node="tools",
)
```

### Edit — update AIMessage in state

When edited, the pending `AIMessage` (which holds the tool calls) is replaced with a new one containing the updated args:

```python
updated_msg = AIMessage(
    content=last_msg.content,
    tool_calls=edited_calls,
    id=last_msg.id,
)
agent.update_state(config, {"messages": [updated_msg]})
```

### Persistent Memory — SqliteSaver

Conversation state is persisted across sessions in a local SQLite database (`research_agent.db`), so the agent remembers context even after restarting.

---

## Setup & Run

**Install dependencies:**

```bash
uv add langchain langchain-community langchain-ollama langgraph-checkpoint-sqlite ollama arxiv ddgs wikipedia python-dotenv
```

**Pull the local model:**

```bash
ollama pull qwen3.5:9b
```

**Run the agent:**

```bash
uv run main.py
```

**Exit:**

```
You: quit
```

---

## Example Session

```
Research Agent Pro — Human-in-the-Loop Edition
==================================================
At each tool call you can: (a)pprove  (e)dit  (r)eject
==================================================

You: search wikipedia for python programming language
Calling tools: ['wikipedia']

[HITL Checkpoint]
  Tool : wikipedia
  Args : {'query': 'python programming language'}
  Options: (a)pprove  (e)dit  (r)eject
  Decision: e
  Edit args (press Enter to keep current value):
    query [python programming language]: Python language history
  -> 'wikipedia' will run with updated args.

Agent: Python is a high-level, general-purpose programming language...
```
