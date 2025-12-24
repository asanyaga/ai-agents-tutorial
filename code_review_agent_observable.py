import tiktoken
import json

from typing import Callable, Dict
import openai
import os

from typing import List,Optional
import uuid


import json
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class AgentLogger:
    """Structured logging for agent actions"""
    def __init__(self, agent_id: str = "agent-1"):
        self.agent_id = agent_id
        self.logs = []

    def log(self, level: LogLevel, event_type: str, message: str, 
            metadata: Optional[dict[str, Any]] = None):
        """Create a structured log entry"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "level": level.value,
            "event_type": event_type,
            "message": message,
            "metadata": metadata or {}
        }
        self.logs.append(log_entry)

        # Also print for real-time feedback
        print(f"[{level.value}] {event_type}: {message}")
    
    def get_logs(self, event_type: Optional[str] = None) -> list:
        """Retrieve logs optionally filtered by event type"""
        if event_type:
            return [log for log in self.logs if log["event_type"] == event_type]
        return self.logs

    def save_logs(self, file_path: str):
        """Persist logs to a JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.logs, f, indent=2)

class Span:
    """Represents a single unit of work in a trace"""

    def __init__(self, name: str, span_type: str, parent_id: Optional[str] = None):
        self.span_id = str(uuid.uuid4())[:8]
        self.parent_id = parent_id
        self.name = name
        self.span_type = span_type
        self.start_time = time.time()
        self.end_time = None
        self.status = "running"
        self.metadata = {}
        self.children: list[Span] = []

    def end(self, status: str = "success", metadata: Optional[dict] = None):
        """Mark span as complete"""
        self.end_time = time.time()
        self.status = status
        if metadata:
            self.metadata.update(metadata)
    
    def duration_ms(self) -> float:
        """Calculate span duration in milliseconds"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    def add_child(self, child: 'Span'):
        """Add child span"""
        self.children.append(child)

    def to_dict(self) -> dict:
        """Convert span to dict for serialization"""
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "type": self.span_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms(),
            "status": self.status,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children]
        }

class TraceManager:
    """Manages execution traces"""
    def __init__(self):
        self.traces = []
        self.active_spans = {}
        self.current_span_id = None
    
    def start_span(self, name: str, span_type: str) -> str:
        """Create and activate a new span"""
        parent_id = self.current_span_id
        span = Span(name, span_type, parent_id)
        self.active_spans[span.span_id] = span

        if parent_id and parent_id in self.active_spans:
            self.active_spans[parent_id].add_child(span)
        
        self.current_span_id = span.span_id
        return span.span_id
    
    def end_span(self, span_id: str, status: str = "success", 
                 metadata: Optional[dict] = None):
        """Complete a span and update current span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end(status, metadata)

            # Move current span to parent
            if span.parent_id:
                self.current_span_id = span.parent_id
            else:
                # Root span completed - save trace
                self.traces.append(span)
                self.current_span_id = None

    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span"""
        if self.current_span_id:
            return self.active_spans.get(self.current_span_id)
        return None
    
    def save_traces(self, file_path: str):
        """Save all traces to a file"""
        traces_data = [trace.to_dict() for trace in self.traces]
        with open(file_path, "w") as f:
            json.dump(traces_data, f, indent=2)
            
class TraceVisualizer:
    """Generate human readable trace visualizations"""

    @staticmethod
    def format_trace(span: dict, indent: int = 0) -> str:
        """Recursively format a trace and its children"""
        prefix = "  " * indent
        
        duration = span["duration_ms"]
        duration_str = f"{duration:.0f}ms"

        status_icon = f"❌ Status: {span["status"]}" if span["status"] == "ERROR" else f"✅ Status: {span["status"]}" 

        # Build line
        line = f"{prefix}{status_icon} {span['name']} ({span['type']}) - {duration_str}"

        if span.get("metadata"):
            metadata = span["metadata"]
            if "cost_usd" in metadata:
                line += f" [${metadata['cost_usd']:.4f}]"
            if "error" in metadata:
                line += f" [ERROR: {metadata['error']}]"

        lines = [line]

        # Recursively format children
        for child in span.get("children", []):
            lines.append(TraceVisualizer.format_trace(child, indent + 1))
        
        return "\n".join(lines)
    
    @staticmethod
    def print_all_traces(traces: List[dict]):
        """Print all traces in a readable format"""
        print("\n" + "=" * 60)
        print("EXECUTION TRACES")
        print("=" * 60)

        for i, trace in enumerate(traces, 1):
            print(f"Trace {i}:")
            print(TraceVisualizer.format_trace(trace))

class TokenTracker:
    """Track token usage and estimate costs"""

    # Pricing per 1M tokens 
    PRICING = {
        "gpt-4.1": {"input": 2.50, "output": 10.00},
        "gpt-4.1-mini": {"input": 0.15, "output": 0.60}
    }

    def __init__(self, model: str):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.token_log = []
    
    def track_usage(self, input_tokens: int, output_tokens: int, 
                    operation: str = "llm_call"):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": self._calculate_cost(input_tokens, output_tokens)
        }
        self.token_log.append(entry)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        if self.model not in self.PRICING:
            return 0.0
        pricing = self.PRICING[self.model]
        input_cost = (input_tokens / 1000000) * pricing["input"]
        output_cost = (output_tokens / 1000000) * pricing["output"]
        return input_cost + output_cost
    
    def get_summary(self) -> dict:
        """Get usage summary"""
        return {
            "model": self.model,
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": self._calculate_cost(
                self.total_input_tokens, self.total_output_tokens
            )
        }

from datetime import datetime
from typing import List
class MetricsCollector:
    """Collect and analyze performance metrics"""

    def __init__(self):
        self.metrics = {
            "iteration_count": 0,
            "tool_calls": {},  # tool_name: count
            "tool_latencies": {},  # tool_name: durations
            "llm_latencies": [],
            "errors": [],
            "loop_detection": []  # Track repeated tool calls
        }
        self.last_n_tools = []  # Sliding window for loop detection
    
    def record_iteration(self):
        """Increment iteration counter"""
        self.metrics["iteration_count"] += 1
    
    def record_tool_call(self, tool_name: str, duration_ms: float):
        """Record a tool invocation"""
        if tool_name not in self.metrics["tool_calls"]:
            self.metrics["tool_calls"][tool_name] = 0
            self.metrics["tool_latencies"][tool_name] = []
        
        self.metrics["tool_calls"][tool_name] += 1
        self.metrics["tool_latencies"][tool_name].append(duration_ms)

        # Loop detection: track last 5 tool calls
        self.last_n_tools.append(tool_name)
        if len(self.last_n_tools) > 5:
            self.last_n_tools.pop(0)
        
        # Check for repeated patterns
        if len(self.last_n_tools) == 5:
            if len(set(self.last_n_tools)) <= 2:  # 1 or 2 unique tool calls in last 5
                self.metrics["loop_detection"].append({
                    "iteration": self.metrics["iteration_count"],
                    "pattern": self.last_n_tools.copy()
                })
    
    def record_llm_latency(self, duration_ms: float):
        """Record LLM call duration"""
        self.metrics["llm_latencies"].append(duration_ms)
    
    def record_error(self, error_type: str, details: str):
        """Record an error"""
        self.metrics["errors"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": error_type,
            "details": details
        })
    
    def get_summary(self) -> dict:
        """Generate metrics summary"""
        summary = {
            "total_iterations": self.metrics["iteration_count"],
            "total_tool_calls": sum(self.metrics["tool_calls"].values()),
            "tool_usage": self.metrics["tool_calls"],
            "error_count": len(self.metrics["errors"]),
            "potential_loops": len(self.metrics["loop_detection"])
        }

        # Calculate average latencies
        if self.metrics["llm_latencies"]:
            summary["avg_llm_latency_ms"] = (
                sum(self.metrics["llm_latencies"]) / len(self.metrics["llm_latencies"])
            )

        summary["tool_avg_latencies"] = {}
        for tool, latencies in self.metrics["tool_latencies"].items():
            if latencies:
                summary["tool_avg_latencies"][tool] = sum(latencies) / len(latencies)
        
        return summary
    
    def check_anomalies(self) -> List[str]:
        """Detect anomalous behavior"""
        warnings = []
        
        # Check for excessive iterations
        if self.metrics["iteration_count"] > 15:
            warnings.append(f"High iteration count: {self.metrics['iteration_count']}")
        
        # Check for tool call loops
        if self.metrics["loop_detection"]:
            warnings.append(
                f"Possible loop detected: {len(self.metrics['loop_detection'])} instances"
            )
        
        # Check for excessive errors
        if len(self.metrics["errors"]) > 3:
            warnings.append(f"Multiple errors: {len(self.metrics['errors'])}")
        
        # Check for slow operations
        if self.metrics["llm_latencies"]:
            avg_llm = sum(self.metrics["llm_latencies"]) / len(self.metrics["llm_latencies"])
            if avg_llm > 2000:
                warnings.append(f"Slow LLM calls: avg {avg_llm:.0f}ms")
        
        return warnings

## Set up the tools and tools registry
def write_test(file_path:str, test_code: str) -> str:
    """Write test code to a test file"""
    try:
        test_dir = os.path.dirname(file_path) or "tests"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        with open(file_path, "w") as f:
            f.write(test_code)
        return f"Test file created: {file_path}"
    except Exception as e:
        return f"Error writing test file {file_path: {e}}"

def run_test(file_path: str) -> str:
    """Run a Python test file and return results"""
    try:
        import subprocess
        result = subprocess.run(
            ["python","-m","pytest", file_path,"-v"],
            capture_output=True,
            text=True,
            timeout=30
        )
        return f"Exit code {result.returncode}\n\nOuput:\n{result.stdout}\n\nErrors:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Test execution timed out after 30 seconds"
    except Exception as e:
        return f"Error running tests: {e}"

def read_file(file_path: str) -> str:
    """Read contents of a Python file"""
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    
    with open(file_path, "r") as f:
        return f.read()
    
def print_review(review: str):
    print(f"Review: {review}")
    return f"Printed review: {review}"

def patch_file(filepath: str, content: str) -> str:
    """Writes the given content to a file, completely replacing its current content."""
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"File successfully updated: {filepath}. New content written."
    except Exception as e:
        return f"Error writing to file {filepath}: {e}"
        
class ToolRegistry:
    """Holds available tools and dispatches them by name."""
    def __init__(self):
        self.tools: Dict[str,Callable] = {}
    
    def register(self, name:str, func: Callable):
        self.tools[name] = func

    def call(self, name:str, *args, **kwargs):
        if name not in self.tools:
            return f"Unknown tool: {name}"
        return self.tools[name](*args, **kwargs)

class CodeReviewAgentObservable:
    def __init__(self, tools_registry: ToolRegistry, model="gpt-4.1",
                 memory_file="agent_memory.json", summarize_after=10,
                 max_context_tokens=6000):
        self.tools = tools_registry
        self.model = model
        self.conversation_history = []  # Short-term memory
        self.memory_file = memory_file
        self.load_long_term_memory()  # Long-term memory (key-value store)
        self.conversation_summary = ""  # Summarized conversation history
        self.summarize_after = summarize_after
        self.turns_since_summary = 0
        self.max_context_tokens = max_context_tokens
        
        # Planning-specific attributes
        self.current_plan = []  # List of planned steps
        self.completed_steps = []  # Track what has been done
        self.plan_created = False

        # Observability components
        self.logger = AgentLogger(agent_id=f"code-review-{int(time.time())}")
        self.logger.log(LogLevel.INFO, "AGENT_INIT", "Agent initialized",
                       {"model": model, "max_tokens": max_context_tokens})
        self.tracer = TraceManager()
        self.token_tracker = TokenTracker(model=model)
        self.metrics = MetricsCollector()

        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string"""
        return len(self.tokenizer.encode(text))
    
    def trim_history_to_fit(self, system_message: str):
        """Remove old messages until we fit within the token budget"""
        fixed_tokens = self.count_tokens(system_message)
        history_tokens = sum([self.count_tokens(msg["content"]) 
                            for msg in self.conversation_history])
        total_tokens = fixed_tokens + history_tokens

        while total_tokens > self.max_context_tokens and len(self.conversation_history) > 2:
            removed_msg = self.conversation_history.pop(0)
            total_tokens -= self.count_tokens(removed_msg["content"])

        return total_tokens

    def summarize_history(self):
        """Use LLM to summarize the conversation so far."""
        if len(self.conversation_history) < 3:
            return
        
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.conversation_history
        ])

        summary_prompt = f"""Summarize this conversation in 3-4 sentences,
        preserving key facts, decisions, and actions taken:
        {history_text}

        Previous Summary: {self.conversation_summary or 'None'}
        """

        response = openai.responses.create(
            model=self.model, 
            input=[{"role": "user", "content": summary_prompt}]
        )
        self.conversation_summary = response.output_text

        # Keep only the last few turns + the summary
        recent_turns = self.conversation_history[-4:]
        self.conversation_history = recent_turns
        self.turns_since_summary = 0

    def remember(self, key: str, value: str):
        """Store information in long term memory."""
        self.long_term_memory[key] = value
        self.save_long_term_memory()
    
    def recall(self, key: str) -> str:
        """Retrieve information from long term memory"""
        return self.long_term_memory.get(key, "No memory found for this key.")
    
    def get_relevant_memories(self) -> str:
        """Format long term memories for inclusion in prompts."""
        if not self.long_term_memory:
            return "No stored memories"
        
        memories = "\n".join([f"- {k}: {v}" for k, v in self.long_term_memory.items()])
        return f"Relevant memories:\n{memories}"
    
    def save_long_term_memory(self):
        """Persist long term memory to JSON file"""
        try:
            with open(self.memory_file, "w") as f:
                json.dump(self.long_term_memory, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save memory to {self.memory_file}: {e}")

    def load_long_term_memory(self):
        """Load long term memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.long_term_memory = json.load(f)
                print(f"Loaded {len(self.long_term_memory)} memories from {self.memory_file}")
            except Exception as e:
                print(f"Warning: Could not load memory from {self.memory_file}: {e}")
                self.long_term_memory = {}
        else:
            self.long_term_memory = {}
    
    def create_plan(self, user_query: str) -> list:
        """Generate a step by step plan for the user's request"""
        planning_prompt = f"""
        Given this task: "{user_query}"
        Create a detailed execution plan with numbered steps. 
        Each step should be a specific action.

        Available tools:
        - read_file(filepath): Read a file's contents
        - patch_file(filepath, content): Update a file
        - write_test(file_path, test_code): Create a test file
        - run_test(file_path): Execute tests

        Format your response as a JSON list of steps:
        [
            {{"step": 1, "action": "description", "tool": "tool_name"}},
            {{"step": 2, "action": "description", "tool": "tool_name"}}
        ]

        Only include necessary steps. Be specific about which files to work with.
        Respond with ONLY the JSON array—no markdown, no extra text.
        """

        response = openai.responses.create(
            model=self.model,
            input=[{"role": "user", "content": planning_prompt}]
        )
        
        try:
            plan = json.loads(response.output_text)
            self.current_plan = plan
            self.plan_created = True
            return plan
        except json.JSONDecodeError:
            self.current_plan = [
                {"step": 1, "action": "Proceed step by step", "tool": "read_file"}
            ]
            self.plan_created = True
            return self.current_plan
    
    def _build_plan_context(self, next_step) -> str:
        """Format plan information for the prompt"""
        completed = "\n".join([
            f"  ✓ Step {step['step']}: {step['action']}" 
            for step in self.completed_steps
        ])

        if next_step:
            current = f"\n→ CURRENT: Step {next_step['step']}: {next_step['action']}"
        else:
            current = "\n→ All steps completed"
        
        remaining = "\n".join([
            f"  Step {step['step']}: {step['action']}" 
            for step in self.current_plan[len(self.completed_steps)+1:]
        ])

        return f"""
## Execution Plan Progress

Completed:
{completed if completed else "  None yet"}
{current}

Remaining:
{remaining if remaining else "  None"}
"""

    def build_system_prompt(self, plan_context: str = "") -> str:
        """Construct the ReAct system prompt with current context and plan."""
        return f"""You are a code review assistant using the ReAct pattern with planning.

## Available Tools
- read_file(filepath): Read contents of a file
- patch_file(filepath, content): Replace file contents entirely
- write_test(file_path, test_code): Create a test file
- run_test(file_path): Execute tests and return results

## Context
{self.get_relevant_memories()}

Conversation summary: {self.conversation_summary or 'This is the start of the conversation.'}

{plan_context}

## Response Format

You MUST respond with valid JSON in one of these two formats:

### Format 1: When you need to use a tool
{{
    "thought": "Your reasoning about what to do and why",
    "action": {{
        "tool": "tool_name",
        "args": ["arg1", "arg2"]
    }}
}}

### Format 2: When the task is complete
{{
    "thought": "Your reasoning about why the task is complete",
    "answer": "Your final response to the user"
}}

## Rules
1. Always include "thought" explaining your reasoning
2. Include "action" when you need to call a tool
3. Include "answer" only when ALL plan steps are complete
4. Never include both "action" and "answer"
5. Respond with ONLY valid JSON—no markdown, no extra text
6. Follow the execution plan systematically
7. After each successful action, the system will mark that step complete

## Example

User: Review auth.py and fix any bugs

Response 1:
{{"thought": "I need to read the file first to see its contents.", "action": {{"tool": "read_file", "args": ["auth.py"]}}}}

Observation: def check(u): return db.user = u

Response 2:
{{"thought": "There's a bug: using = (assignment) instead of == (comparison). I'll fix it.", "action": {{"tool": "patch_file", "args": ["auth.py", "def check(u): return db.user == u"]}}}}

Observation: File successfully updated: auth.py

Response 3:
{{"thought": "The bug is fixed. The comparison operator is now correct.", "answer": "Fixed auth.py: changed assignment operator (=) to comparison operator (==) in the return statement."}}
"""

    def think(self, user_input: str):
        """LLM decides which tool to use with plan awareness and observability."""
        self.logger.log(LogLevel.INFO, "THINK_START", "Starting Reasoning",
                       {"user_input": user_input[:100]})
        think_span_id = self.tracer.start_span("Think", "LLM_CALL")
        
        try:
            # First request: create a plan
            if not self.plan_created:
                plan = self.create_plan(user_query=user_input)
                plan_summary = "\n".join([
                    f"Step {step['step']}: {step['action']}" 
                    for step in plan
                ])
                
                # Return a special response indicating the plan was created
                return json.dumps({
                    "thought": "I've analyzed the task and created an execution plan.",
                    "plan_created": True,
                    "plan": plan_summary
                })

            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.turns_since_summary += 1

            # Check if we should summarize
            if self.turns_since_summary >= self.summarize_after:
                self.summarize_history()

            # Get current step from plan
            next_step = None
            if len(self.completed_steps) < len(self.current_plan):
                next_step = self.current_plan[len(self.completed_steps)]
            
            # Build context with plan information
            plan_context = self._build_plan_context(next_step)

            # Include long term memory & summary in system context
            system_message_context = self.build_system_prompt(plan_context)

            self.trim_history_to_fit(system_message_context)
            
            # Build prompt with system instructions
            messages = [
                {"role": "system", "content": system_message_context}
            ] + self.conversation_history

            # Count input tokens
            input_text = json.dumps([msg["content"] for msg in messages])
            input_tokens = self.count_tokens(input_text)

            start_time = time.time()
            response = openai.responses.create(model=self.model, input=messages)
            duration_ms = (time.time() - start_time) * 1000

            decision = response.output_text

            # Add assistant's decision to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": decision
            })

            # Count output tokens and track usage
            output_tokens = self.count_tokens(decision)
            self.token_tracker.track_usage(input_tokens, output_tokens, "think")
            self.metrics.record_llm_latency(duration_ms)

            self.tracer.end_span(think_span_id, "SUCCESS", {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": self.token_tracker._calculate_cost(input_tokens, output_tokens)
            })
            self.logger.log(LogLevel.INFO, "THINK_COMPLETE", "Reasoning Complete",
                           {"decision": decision[:200]})

            return decision
            
        except Exception as e:
            self.tracer.end_span(think_span_id, "ERROR", {"error": str(e)})
            self.logger.log(LogLevel.ERROR, "THINK_ERROR", f"Think failed: {str(e)}",
                           {"error": str(e)})
            raise
    
    def act(self, action: dict):
        """Execute the chosen tool and update plan progress with observability."""
        self.logger.log(LogLevel.INFO, "ACT_START", "Executing action",
                       {"tool": action.get("tool"), "args": action.get("args", [])})
        act_span_id = self.tracer.start_span("Act", "TOOL_EXECUTION")

        try:
            tool_name = action.get("tool")
            args = action.get("args", [])

            # Create nested span for the specific tool
            tool_span_id = self.tracer.start_span(f"Tool: {tool_name}", "TOOL_CALL")

            self.logger.log(LogLevel.DEBUG, "TOOL_CALL", f"Calling {tool_name}",
                           {"tool": tool_name, "args": args})

            start_time = time.time()
            result = self.tools.call(tool_name, *args)
            duration_ms = (time.time() - start_time) * 1000

            self.tracer.end_span(tool_span_id, "SUCCESS",
                               {"tool": tool_name, "result_size": len(str(result))})

            self.logger.log(LogLevel.INFO, "TOOL_COMPLETE", f"{tool_name} completed",
                           {"tool": tool_name, "duration_ms": duration_ms})
            self.metrics.record_tool_call(tool_name, duration_ms)

            # Mark current step as complete
            if len(self.completed_steps) < len(self.current_plan):
                current_step = self.current_plan[len(self.completed_steps)]
                self.completed_steps.append(current_step)

            self.conversation_history.append({"role": "system", "content": result})
            self.tracer.end_span(act_span_id, "SUCCESS")
            return result
            
        except Exception as e:
            error_msg = f"Error executing tool: {e}"
            self.logger.log(LogLevel.ERROR, "ACT_ERROR", error_msg,
                           {"action": action, "error": str(e)})
            self.tracer.end_span(act_span_id, "ERROR", {"error": str(e)})
            self.metrics.record_error("TOOL_ERROR", str(e))
            self.conversation_history.append({
                "role": "system",
                "content": error_msg
            })
            return error_msg

    def run(self, user_query: str, max_iterations=10):
        """
        Main execution loop with planning, ReAct pattern, and full observability.
        
        Args:
            user_query: The user's request
            max_iterations: Maximum number of think-act cycles
        
        Returns:
            Final response string
        """
        run_span_id = self.tracer.start_span(
            f"Agent Run: {user_query[:50]}", 
            span_type="AGENT_RUN"
        )

        try:
            current_input = user_query

            for step in range(max_iterations):
                # Create a span for each iteration
                iter_span_id = self.tracer.start_span(
                    name=f"Iteration {step + 1}",
                    span_type="ITERATION"
                )
                self.metrics.record_iteration()

                print(f"\n{'-'*60}")
                print(f"Step {step+1} of {max_iterations}")
                print(f"{'-'*60}")

                llm_response = self.think(current_input)
                print(f"\nAgent's LLM Response:\n{llm_response}")

                # Parse the JSON response
                try:
                    parsed_response = json.loads(llm_response)
                except json.JSONDecodeError as e:
                    print(f"\nJSON Parse Error: {e}")
                    self.metrics.record_error("JSON_PARSE_ERROR", str(e))
                    current_input = (
                        f"Your response was not valid JSON. Error: {e}\n"
                        f"Respond with ONLY valid JSON matching the required format."
                    )
                    self.tracer.end_span(iter_span_id, "JSON_ERROR")
                    continue

                # Handle plan creation response
                if parsed_response.get("plan_created"):
                    print(f"\nPlan Created:")
                    print(parsed_response.get("plan", ""))
                    current_input = "Proceed with step 1 of the plan."
                    self.tracer.end_span(iter_span_id, "PLAN_CREATED")
                    continue

                # Print thought if present
                if "thought" in parsed_response:
                    print(f"\nThought: {parsed_response['thought']}")

                # Check for final answer
                if "answer" in parsed_response:
                    print(f"\nAnswer: {parsed_response['answer']}")
                    print(f"\nProgress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
                    
                    self.tracer.end_span(iter_span_id, "SUCCESS",
                                        {"final_answer": parsed_response["answer"][:100]})
                    self.tracer.end_span(run_span_id, "SUCCESS",
                                        {"total_iterations": step + 1})
                    return parsed_response["answer"]
                
                # Execute action if present
                if "action" in parsed_response:
                    action = parsed_response["action"]
                    tool_name = action.get("tool", "unknown")
                    args = action.get("args", [])

                    print(f"\nAction: {tool_name}({', '.join(repr(a) for a in args)})")
                    
                    observation = self.act(action)
                    
                    # Truncate long observations for display
                    obs_display = observation[:500] + "..." if len(str(observation)) > 500 else observation
                    print(f"\nObservation: {obs_display}")
                    
                    current_input = f"Observation: {observation}"
                    self.tracer.end_span(iter_span_id, "SUCCESS")
                else:
                    # Neither action nor answer
                    print("\nResponse missing both 'action' and 'answer'")
                    current_input = (
                        "Your response must include either 'action' (to use a tool) "
                        "or 'answer' (if the task is complete). Please try again."
                    )
                    self.tracer.end_span(iter_span_id, "INVALID_RESPONSE")

            self.tracer.end_span(run_span_id, "MAX_ITERATIONS",
                               {"completed_steps": len(self.completed_steps),
                                "total_steps": len(self.current_plan)})
            
            # Check for anomalies at the end
            warnings = self.metrics.check_anomalies()
            if warnings:
                print("\n⚠️ Performance Warnings:")
                for warning in warnings:
                    print(f"  {warning}")

            print(f"\nMaximum Iterations ({max_iterations}) reached")
            print(f"Progress: {len(self.completed_steps)}/{len(self.current_plan)} steps completed")
            return "Task Incomplete: Max steps reached"
            
        except Exception as e:
            self.tracer.end_span(run_span_id, "ERROR", {"error": str(e)})
            self.metrics.record_error("RUNTIME_ERROR", str(e))
            raise
        finally:
            self.save_instrumentation()

    def save_instrumentation(self, trace_file="traces.json", log_file="log.json",
                             token_file="tokens.json", metrics_file="metrics.json"):
        """Save all observability data to files."""
        self.tracer.save_traces(trace_file)
        self.logger.save_logs(log_file)

        with open(token_file, "w") as tf:
            json.dump({
                "summary": self.token_tracker.get_summary(),
                "detailed_log": self.token_tracker.token_log
            }, tf, indent=2)
        
        with open(metrics_file, "w") as mf:
            json.dump({
                "summary": self.metrics.get_summary(),
                "detailed_metrics": self.metrics.metrics,
                "anomalies": self.metrics.check_anomalies()
            }, mf, indent=2)
        
        print(f"\n📊 Instrumentation saved:")
        print(f"  - Traces: {trace_file}")
        print(f"  - Logs: {log_file}")
        print(f"  - Tokens: {token_file}")
        print(f"  - Metrics: {metrics_file}")

        # Print summary to console
        print(f"\n📈 Execution Summary")
        token_summary = self.token_tracker.get_summary()
        print(f"  Cost: ${token_summary['estimated_cost_usd']:.4f}")
        print(f"  Tokens: {token_summary['total_tokens']:,}")
        metric_summary = self.metrics.get_summary()
        print(f"  Tool calls: {metric_summary['total_tool_calls']}")
        print(f"  Iterations: {metric_summary['total_iterations']}")
    
    def print_trace_summary(self):
        """Print a visual summary of execution traces"""
        traces = [trace.to_dict() for trace in self.tracer.traces]
        TraceVisualizer.print_all_traces(traces)