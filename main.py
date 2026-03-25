"""
Assignment 04 — Research Agent Pro: Human-in-the-Loop (HITL)

PROBLEM STATEMENT:
Research Agent Pro takes an already-functional multi-tool research agent and
elevates it from a fully autonomous system to a human-governed agentic AI.
The core upgrade is the integration of Human-in-the-Loop (HITL) middleware,
which introduces approval gates, edit capabilities, and rejection controls
directly into the agent's reasoning-action loop. This mirrors how production
AI systems operate in enterprise environments: the agent proposes actions,
the human disposes, and the graph orchestrates the handoff transparently.
The result is an agent that is both powerful and trustworthy.

HITL Execution Flow:
  1. User sends query              — Research question enters the agent
  2. Agent reasons & selects tool  — LLM analyzes via ReAct loop, picks best tool
  3. INTERRUPT                     — State saved to SqliteSaver, awaits human
  4. Human reviews tool call       — Tool name, args, and context displayed
  5. Decision: approve/edit/reject — Three-way control over every tool call
  6. Command(resume=...) restores graph — Execution continues from checkpoint
  7. Tool executes (or skips)      — Approved/edited tool runs; rejected is skipped
  8. Agent continues or answers    — LLM processes result, answers or picks next tool

Technology Stack:
  Framework    : LangChain 1.x
  Graph engine : LangGraph + SqliteSaver
  LLM          : Ollama (local)
  Tools        : DuckDuckGo, Arxiv, Wikipedia
  Interface    : CLI + real-time streaming
"""

import sqlite3
from datetime import datetime

# LangChain Agent & Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    wrap_tool_call,
)
from langchain.tools import tool

# LangChain Community
from langchain_community.tools import (
    ArxivQueryRun,
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
)
from langchain_community.utilities import (
    ArxivAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Model
from langchain_ollama import ChatOllama

# LangGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

# --- RESEARCH TOOLS ---
# Web search tool - searches the web for current information
ddgs_warpper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_warpper,
    name="web_search",
    description="Search the web using DuckDuckGo for current information, news, and general web content. Use this when you need up-to-date information or content not available on Wikipedia.",
)

# Wikipedia tool - queries Wikipedia for encyclopedia-style information
wiki_warpper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_warpper,
    name="wikipedia",
    description="Search Wikipedia for encyclopedia-style information, facts, and summaries. Use this for quick factual queries and well-established knowledge.",
)

# arXiv tool - searches for academic papers and research
arxiv_warpper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_warpper,
    name="arxiv",
    description="Search arXiv for academic papers, scientific research, and scholarly articles. Use this for technical and academic research queries.",
)


@tool
def get_current_datetime():
    """Get the current date and time.

    Returns:
        str: The current datetime formatted as a string.
    """
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")


tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = """You are a Research AI Agent, an intelligent assistant specialized in conducting research and gathering information from multiple sources.

Your capabilities:
- Web Search: Use DuckDuckGo to find current information, news, and web content
- Wikipedia: Query for encyclopedia-style facts and well-established knowledge
- arXiv: Search for academic papers and scientific research
- DateTime: Get the current date and time when needed

Guidelines:
1. Always use the most appropriate tool for the type of information needed
2. For factual queries, start with Wikipedia
3. For current events or recent information, use web search
4. For academic or technical research, use arXiv
5. Synthesize information from multiple sources when possible
6. Provide clear, well-structured responses with proper citations
7. If a query is ambiguous, ask for clarification before searching

When responding, cite your sources and provide accurate, up-to-date information.
"""


@wrap_tool_call
def tool_handle_error(request, handler):
    try:
        return handler(request)
    except Exception as err:
        print(f"Error: {err}")


tool_retry = ToolRetryMiddleware(
    max_retries=2,
    max_delay=60,
    backoff_factor=1.5,
    tools=["web_search", "arxiv"],
    on_failure="continue",
)

model_retry = ModelRetryMiddleware(
    max_retries=3,
    on_failure="continue",
    max_delay=60,
    backoff_factor=1.5,
)

tool_call_limit = ToolCallLimitMiddleware(run_limit=5)

human_in_loop = HumanInTheLoopMiddleware(
    interrupt_on={
        "web_search": {"ask_permission_to_search_web": True},
        "wikipedia": {"ask_permission_to_search_wikipedia": True},
        "arxiv": {"ask_permission_to_search_arxiv": True},
        "get_current_datetime": {"ask_permission_to_read_current_date_time": True},
    }
)


def create_reasearch_agent():
    """Create and configure the research agent with LLM, tools, and memory.

    Returns:
        Configured LangChain agent with web search, Wikipedia, and arXiv tools.
    """
    llm = ChatOllama(model="qwen3.5:9b", temperature=0)

    conn = sqlite3.connect("research_agent.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    middleware = [tool_handle_error, tool_retry]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        checkpointer=memory,
        middleware=middleware,
        interrupt_before=["tools"],
        name="research_agent",
    )

    return agent


def banner():
    """Display the agent banner."""
    print("Research Agent Pro — Human-in-the-Loop Edition")
    print("=" * 50)
    print("At each tool call you can: (a)pprove  (e)dit  (r)eject")
    print("=" * 50)


def stream_response(agent, query: str, config: dict):
    """Stream and print the agent's response to a query.

    Handles human-in-the-loop interrupts: pauses before each tool call,
    shows tool name and args, then gives three options — approve, edit,
    or reject — before resuming or skipping the tool.

    Args:
        agent: The LangChain agent instance.
        query: The user's query string.
        config: Configuration dictionary with thread_id for memory.
    """
    current_input = {"messages": [HumanMessage(content=query)]}

    while True:
        # Stream until the graph finishes or hits an interrupt
        for chunk in agent.stream(current_input, config=config, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                if isinstance(latest_message, AIMessage):
                    print(f"Agent: {latest_message.content}")
            elif hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                print(
                    f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}"
                )

        # After stream ends, check if the graph paused before tools
        state = agent.get_state(config)
        if not state.next:
            # No more steps — fully done
            break

        # Interrupted before tools node — show HITL options for each pending tool call
        last_msg = state.values["messages"][-1]
        if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
            break

        # Track edits across all tool calls
        edited_calls = [dict(tc) for tc in last_msg.tool_calls]
        any_edited = False

        for i, tc in enumerate(last_msg.tool_calls):
            tool_name = tc["name"]
            tool_args = tc.get("args", {})

            print("\n[HITL Checkpoint]")
            print(f"  Tool : {tool_name}")
            print(f"  Args : {tool_args}")
            print("  Options: (a)pprove  (e)dit  (r)eject")
            choice = input("  Decision: ").strip().lower()

            if choice in ("r", "reject"):
                # Inject a ToolMessage so the agent knows the tool was skipped
                reason = input("  Reason for rejection: ").strip() or "Rejected by user"
                agent.update_state(
                    config,
                    {
                        "messages": [
                            ToolMessage(
                                content=f"Tool '{tool_name}' was rejected by user. Reason: {reason}",
                                tool_call_id=tc["id"],
                            )
                        ]
                    },
                    as_node="tools",
                )
                print(f"  -> '{tool_name}' rejected. Agent will be notified.")

            elif choice in ("e", "edit"):
                # Let the user modify each argument before the tool runs
                print("  Edit args (press Enter to keep current value):")
                new_args = {}
                for key, val in tool_args.items():
                    new_val = input(f"    {key} [{val}]: ").strip()
                    new_args[key] = new_val if new_val else val
                edited_calls[i] = {**tc, "args": new_args}
                any_edited = True
                print(f"  -> '{tool_name}' will run with updated args.")

            else:
                print(f"  -> '{tool_name}' approved.")

        # If any tool args were edited, update the AIMessage in state
        if any_edited:
            updated_msg = AIMessage(
                content=last_msg.content,
                tool_calls=edited_calls,
                id=last_msg.id,
            )
            agent.update_state(config, {"messages": [updated_msg]})

        # Resume graph — tools run (or are skipped if rejected)
        current_input = Command(resume=True)


def main():
    """Main entry point for the Research Agent Pro.

    Runs an interactive CLI session where users can query the research agent.
    Every tool call is intercepted for human review before execution.
    """
    banner()
    agent = create_reasearch_agent()
    config = {"configurable": {"thread_id": "research-session-1"}}

    while True:
        try:
            query = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodBye! Happy Researching!")
            break

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


main()
