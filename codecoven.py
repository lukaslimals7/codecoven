import subprocess
import uuid
import ollama
from langchain_ollama import ChatOllama
from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
import sys
import json
import platform

# Initialize Rich console
console = Console()

# Initialize Ollama LLM with a model suitable for tool calling
try:
    llm = ChatOllama(model="qwen2.5:3b")
except Exception as e:
    console.print(f"[red]Error initializing LLM: {e}[/red]")
    sys.exit(1)

# Define AI Agent State
class State(TypedDict):
    messages: Annotated[list, ...]
    tool_calls: List[Dict[str, Any]]

# Define tools
@tool
def run_shell_command(command: str) -> str:
    """Execute a shell command and return its output if safe."""
    try:
        # Enhanced command validation to prevent dangerous commands
        dangerous_commands = ["rm -rf", "mkfs", "dd if=", ":(){ :|: & };:",
                            "reboot", "halt", "shutdown", "> /dev/", "chmod -R 777"]
        if any(dangerous in command.lower() for dangerous in dangerous_commands):
            return "[red]Error: Potentially dangerous command detected. Execution aborted.[/red]"
        # Adjust command for platform-specific cases
        if command.lower().startswith("open ") and platform.system() == "Linux":
            command = command.replace("open -a", "").strip()
            command = command.replace("open", "").strip()
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=True
        )
        return result.stdout.strip() if result.stdout else "Command executed successfully."
    except subprocess.CalledProcessError as e:
        return f"[red]Error executing command: {e}\n{e.stderr}[/red]"
    except Exception as e:
        return f"[red]Unexpected error: {e}[/red]"

@tool
def run_nmap_scan(target: str, options: str = "-sP") -> str:
    """Run an nmap scan on a target with specified options."""
    try:
        # Validate target to prevent command injection
        if not all(c.isalnum() or c in ".-/" for c in target):
            return "[red]Error: Invalid target for nmap scan.[/red]"
        command = f"nmap {options} {target}"
        return run_shell_command.invoke(command)
    except Exception as e:
        return f"[red]Error running nmap scan: {e}[/red]"

@tool
def run_nikto_scan(target: str, options: str = "-h") -> str:
    """Run a Nikto web vulnerability scan on a target with specified options."""
    try:
        # Validate target to prevent command injection
        if not (target.startswith("http://") or target.startswith("https://")):
            return "[red]Error: Target must start with http:// or https:// for Nikto scan.[/red]"
        if not all(c.isalnum() or c in ".-/:?=&" for c in target):
            return "[red]Error: Invalid target for Nikto scan.[/red]"
        command = f"nikto {options} {target}"
        return run_shell_command.invoke(command)
    except Exception as e:
        return f"[red]Error running Nikto scan: {e}[/red]"

@tool
def run_sqlmap_scan(target: str, options: str = "--batch --level=1") -> str:
    """Run a SQLMap scan to detect SQL injection vulnerabilities on a target with specified options."""
    try:
        # Validate target to prevent command injection
        if not (target.startswith("http://") or target.startswith("https://")):
            return "[red]Error: Target must start with http:// or https:// for SQLMap scan.[/red]"
        if not all(c.isalnum() or c in ".-/:?=&" for c in target):
            return "[red]Error: Invalid target for SQLMap scan.[/red]"
        command = f"sqlmap {options} -u {target}"
        return run_shell_command.invoke(command)
    except Exception as e:
        return f"[red]Error running SQLMap scan: {e}[/red]"

@tool
def analyze_vulnerabilities(tool_output: str) -> str:
    """Analyze the output of a pentesting tool and suggest vulnerabilities and exploits."""
    try:
        prompt = f"""
You are an expert pentesting and CTF assistant. Analyze the following pentesting tool output to identify potential vulnerabilities and suggest possible exploits or commands to further investigate or exploit them. Provide a concise summary of vulnerabilities and a list of suggested commands or exploits.

Tool Output:
{tool_output}

Response format:
- Vulnerabilities: [List vulnerabilities identified]
- Suggested Exploits/Commands: [List commands or exploit suggestions]
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        return f"[red]Error analyzing vulnerabilities: {e}[/red]"

# Bind tools to LLM
tools = [run_shell_command, run_nmap_scan, run_nikto_scan, run_sqlmap_scan, analyze_vulnerabilities]
llm_with_tools = llm.bind_tools(tools)

# Convert natural language to tool call or execute shell command directly
def process_user_input(state: State) -> Dict[str, Any]:
    """Process user input: execute shell commands directly or call a tool via LLM."""
    try:
        user_input = state["messages"][-1].content.strip()
        # Check if input starts with a tool name
        tool_names = [tool.name for tool in tools]
        if any(user_input.startswith(name) for name in tool_names):
            # Route to LLM for tool call processing
            response = llm_with_tools.invoke([HumanMessage(content=user_input)])
            tool_calls = getattr(response, 'tool_calls', [])
            if tool_calls:
                state["tool_calls"] = tool_calls
                return state
            return {"messages": [AIMessage(content=f"[red]Error: Invalid tool call format for {user_input}[/red]")]}
        
        # Treat as a direct shell command
        try:
            output = run_shell_command.invoke(user_input)
            return {
                "messages": [
                    AIMessage(
                        content=f"[cyan]Command:[/cyan] {user_input}\n\n[green]Output:[/green]\n{output}"
                    )
                ]
            }
        except Exception as e:
            # Fallback to LLM for natural language processing
            response = llm_with_tools.invoke([HumanMessage(content=user_input)])
            tool_calls = getattr(response, 'tool_calls', [])
            if tool_calls:
                state["tool_calls"] = tool_calls
                return state
            return {"messages": [AIMessage(content=f"[red]Error: Invalid command and no tool calls generated: {e}[/red]")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"[red]Error processing input: {e}[/red]")]}

# Execute tool calls
def execute_tool_calls(state: State) -> Dict[str, Any]:
    """Execute any tool calls specified in the state."""
    try:
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return state
        
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                result = tool.invoke(tool_args)
                # If the tool is a scanning tool, analyze its output for vulnerabilities
                if tool_name in ["run_nmap_scan", "run_nikto_scan", "run_sqlmap_scan"]:
                    vuln_analysis = analyze_vulnerabilities.invoke(result)
                    results.append(
                        f"[cyan]Tool: {tool_name}[/cyan]\n[green]Result:[/green]\n{result}\n\n[blue]Vulnerability Analysis:[/blue]\n{vuln_analysis}"
                    )
                else:
                    results.append(f"[cyan]Tool: {tool_name}[/cyan]\n[green]Result:[/green]\n{result}")
            else:
                results.append(f"[red]Error: Tool {tool_name} not found.[/red]")
        
        return {"messages": [AIMessage(content="\n\n".join(results))], "tool_calls": []}
    except Exception as e:
        return {"messages": [AIMessage(content=f"[red]Error executing tool calls: {e}[/red]")], "tool_calls": []}

# Setup Workflow
try:
    memory = MemorySaver()
    workflow = StateGraph(State)
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("execute_tool_calls", execute_tool_calls)
    workflow.add_edge(START, "process_user_input")
    workflow.add_conditional_edges(
        "process_user_input",
        lambda state: "execute_tool_calls" if state.get("tool_calls") else END,
        {"execute_tool_calls": "execute_tool_calls", END: END}
    )
    workflow.add_edge("execute_tool_calls", END)
    graph = workflow.compile(checkpointer=memory)
except Exception as e:
    console.print(f"[red]Error setting up workflow: {e}[/red]")
    sys.exit(1)

# Configuration for conversation thread
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# Create prompt session with history and key bindings
history = InMemoryHistory()
session = PromptSession(
    "User: ",
    history=history,
    enable_history_search=True,
    multiline=False
)

# Key bindings for arrow key navigation
bindings = KeyBindings()

@bindings.add('up')
def _(event):
    event.app.current_buffer.history_backward()

@bindings.add('down')
def _(event):
    event.app.current_buffer.history_forward()

# Main interaction loop with Rich
def run_pentest_agent():
    console.print(Panel.fit("[bold cyan] CodeCoven 🧙🏼 \n Pentest/CTF AI agent [/bold cyan]\nType 'quit' to exit", border_style="green"))
    console.print("[yellow]Available tools: run_shell_command, run_nmap_scan, run_nikto_scan, run_sqlmap_scan, analyze_vulnerabilities[/yellow]")
    while True:
        try:
            user_input = session.prompt(key_bindings=bindings)
            if user_input.lower() in {"q", "quit", "exit"}:
                console.print("[yellow]Goodbye![/yellow]")
                break
            if not user_input.strip():
                console.print("[yellow]Please enter a valid command.[/yellow]")
                continue
            output = graph.invoke({"messages": [HumanMessage(content=user_input)], "tool_calls": []}, config=config)
            console.print(Panel(output["messages"][-1].content, title="AI Response", border_style="blue"))
        except KeyboardInterrupt:
            console.print("[yellow]Operation cancelled by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")

if __name__ == "__main__":
    run_pentest_agent()
