# Agentic Architecture - System Redesign

## Overview

The system has been transformed from a **fixed workflow orchestrator** into a **truly agentic system** where the orchestrator uses LLM reasoning to dynamically decide which tools to call based on the current state and progress toward the goal.

---

## Key Changes

### 1. **Tool-Based Architecture** (`core/tools.py`)

Created a new `tools.py` module that provides:

- **`Tool` class**: Wraps agent functions with metadata (name, description, schema)
- **`ToolSchema` & `ToolParameter`**: Define structured input/output for tools
- **`ToolRegistry`**: Central registry of available tools with execution capabilities

All agents (Planner, Retriever, Analyst, Skeptic, Synthesizer) are now registered as tools that can be called dynamically.

```python
# Example: A tool is registered with schema
@register_tool(
    name="retrieve_evidence",
    description="Retrieve relevant documents from the knowledge base",
    parameters=[
        ToolParameter(name="query", description="Search query", type="string"),
        ToolParameter(name="top_k", description="Number of results", type="number"),
    ],
)
def retrieve_documents(query: str, top_k: int = 6) -> str:
    # Tool implementation
    pass
```

---

### 2. **Structured Skeptic Output** (`agents/skeptic.py` + `core/models.py`)

The Skeptic agent now produces **structured output** with constrained categories, not just free-form text:

#### `SkepticFinding` dataclass:
```python
@dataclass
class SkepticFinding:
    category: str  # hallucination, attack, weak_point, missing_evidence, overconfidence, logical_flaw
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    evidence: str
    recommendation: str
```

#### `StructuredCritique` dataclass:
```python
@dataclass
class StructuredCritique:
    hallucinations: list[SkepticFinding]      # Unsupported claims
    attack_vectors: list[SkepticFinding]       # Ways the answer could be exploited
    weak_points: list[SkepticFinding]          # Fragile arguments/assumptions
    missing_evidence: list[SkepticFinding]     # Key missing information
    overconfidence: list[SkepticFinding]       # Over-certain claims without hedging
    logical_flaws: list[SkepticFinding]        # Reasoning errors, false causality
    overall_confidence: float                   # 0-1 scale
    summary: str
```

**JSON-structured prompting** ensures the LLM returns categorized findings:
```python
# The skeptic uses structured prompting to return JSON
{
  "hallucinations": [{"severity": "high", "description": "..."}],
  "attack_vectors": [...],
  "weak_points": [...],
  "missing_evidence": [...],
  "overconfidence": [...],
  "logical_flaws": [...],
  "overall_confidence": 0.85,
  "summary": "Brief summary of key issues"
}
```

---

### 3. **Enhanced LLM with Tool Calling** (`core/llm.py`)

Added new LLM methods for agentic reasoning:

#### `generate_with_tools()`:
Allows the LLM to request tool calls based on reasoning
```python
response = llm.generate_with_tools(
    prompt="Decide next steps...",
    tools=[tool.to_dict() for tool in registry.get_all()],
)
# Returns: ToolCallResponse with tool_name, arguments, reasoning
```

#### `generate_structured()`:
Returns structured JSON output matching a schema
```python
response = llm.generate_structured(
    prompt="...",
    response_schema={...},
)
```

---

### 4. **Reasoning-Based Orchestrator** (`agents/orchestrator.py`)

The orchestrator is now **agentic** - it reasons about what to do next:

#### Before (Fixed Workflow):
```
Planner → Retriever → Analyst → Skeptic → Synthesizer
```

#### After (Agentic Loop):
```
while iteration < max_iterations:
    # LLM decides which tools to call
    tool_calls = orchestrator._get_next_tools(
        question, objective, iteration_state
    )
    
    # Execute each tool
    for tool_call in tool_calls:
        result = execute_tool(tool_call)
        store(result)
    
    # Check if done
    if has_final_answer:
        break
```

**Key Features**:
- **State-aware**: Tracks what's been completed (retrieved? drafted? critiqued?)
- **Flexible**: Can call tools in any order, skip unnecessary steps
- **Loop-based**: Continues until a stopping condition is met
- **Max iterations**: Safety limit to prevent infinite loops (default: 10)

**Tool Calling Decision Prompt**:
The orchestrator asks the LLM:
> "Based on the current state, which tools should be called next?"
> 
> Current State:
> - Question: "..."
> - Retrieved evidence: YES
> - Has draft: NO
> - Has critique: NO
>
> Available tools: [list of tool schemas]
>
> Return JSON: `[{"tool_name": "analyze_with_evidence", "arguments": {...}}]`

---

### 5. **Tool Implementations** (`core/agent_tools.py`)

Five agent tools wrap the existing agents:

1. **`create_analysis_plan`**
   - Input: `question`, `objective`
   - Output: Analysis plan with goals, requirements, success criteria

2. **`retrieve_evidence`**
   - Input: `query`, `top_k`
   - Output: Summary of relevant documents
   - Side-effect: Stores `RetrievalHit` objects for later use

3. **`analyze_with_evidence`**
   - Input: `question`, `plan`, `memory_context`
   - Output: Draft analysis
   - Uses: Stored retrieved documents from retrieval step

4. **`critique_draft`**
   - Input: `draft`, `question`
   - Output: Structured critique with categories

5. **`synthesize_final_answer`**
   - Input: `draft`, `critique`
   - Output: Final executive-grade answer

---

### 6. **Model Enhancement** (`core/models.py`)

Added new data classes:

- **`ToolCall`**: Records tool invocations with reasoning
- **`SkepticFinding`**: Individual findings from skeptic
- **`StructuredCritique`**: Structured output from skeptic agent
- Updated **`OrchestrationOutput`** to include `tool_calls` trace

```python
@dataclass
class OrchestrationOutput:
    final_answer: str
    citations: list[RetrievalHit]
    critique: str | StructuredCritique  # Now can be structured
    plan: str
    metrics: dict[str, float]
    trace: list[AgentResult]
    tool_calls: list[ToolCall]           # NEW: Track tool invocations
```

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                 AgenticOrchestrator.run()                       │
│                                                                 │
│  1. Input: question, objective                                  │
│  2. Initialize state tracking                                   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  AGENTIC LOOP (iteration < max_iterations)               │  │
│  │                                                          │  │
│  │  _get_next_tools():                                      │  │
│  │    LLM reasons: "What tools do I need now?"              │  │
│  │    Returns: [ToolCall, ToolCall, ...]                   │  │
│  │                                                          │  │
│  │  For each ToolCall:                                      │  │
│  │    - _execute_tool(): Call tool with arguments           │  │
│  │    - Store result in state                               │  │
│  │    - Add to trace                                        │  │
│  │                                                          │  │
│  │  Check: Do we have final_answer? If yes, break          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  3. Return OrchestrationOutput with:                            │
│     - final_answer                                              │
│     - structured_critique                                       │
│     - citations (RetrievalHits)                                 │
│     - tool_calls (trace of decisions)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example Execution Flow

```
Iteration 1:
  State: No evidence, no draft
  → LLM decides: ["create_analysis_plan", "retrieve_evidence"]
  ✓ Plan created
  ✓ Documents retrieved

Iteration 2:
  State: Has evidence, no draft
  → LLM decides: ["analyze_with_evidence"]
  ✓ Draft analysis created

Iteration 3:
  State: Has draft, no critique
  → LLM decides: ["critique_draft"]
  ✓ Structured critique created (hallucinations, weak_points, etc.)

Iteration 4:
  State: Has draft and critique
  → LLM decides: ["synthesize_final_answer"]
  ✓ Final answer created

Iteration 5:
  State: Has final_answer
  → LLM decides: [] (empty - done)
  → Break loop
  → Return results
```

---

## Hallucination & Attack Detection (Skeptic Constraints)

The Skeptic now produces categorized findings to catch:

### **Hallucinations**
- Claims without evidence support
- Contradictions with provided sources
- Fabricated statistics or quotes

### **Attack Vectors**
- Ways the answer could be manipulated
- Exploit vectors in reasoning
- Prompt injection-like weaknesses

### **Weak Points**
- Fragile assumptions
- Single points of failure
- Missing nuance or caveats

### **Missing Evidence**
- Key information gaps
- Unvetted claims
- Sources that should be included

### **Overconfidence**
- Certainty statements without hedging
- Claims beyond available evidence
- Overreaching conclusions

### **Logical Flaws**
- False causality
- Correlation/causation confusion
- Invalid reasoning paths

---

## Benefits of This Architecture

| Aspect | Before | After |
|--------|--------|-------|
| **Workflow** | Fixed, hardcoded | Flexible, reasoning-based |
| **Tool Calls** | Sequential always | Optimized based on state |
| **Skeptic Output** | Unstructured text | Structured with categories |
| **Hallucination detection** | Generic prompting | Specific constraint checking |
| **Decision Tracing** | Implicit | Explicit ToolCall records |
| **Adaptability** | Low | High - LLM decides strategy |
| **Efficiency** | Fixed steps | Optimizable path |

---

## Implementation Notes

### Global State Management
Tools store retrieved `RetrievalHit` objects globally to be reused:
```python
# In agent_tools.py
_last_retrieved_hits: list[RetrievalHit] = []

def retrieve_documents(query: str, top_k: int = 6) -> str:
    global _last_retrieved_hits
    hits = _retriever_agent.run(query, top_k=top_k)
    _last_retrieved_hits = hits
    return format_as_string(hits)
```

### Fallback Mechanisms
- LLM parsing errors → empty tool list (stops gracefully)
- Tool execution fails → error message returned, loop continues
- Structured critique parsing fails → falls back to free-form
- Mock mode when GEMINI_API_KEY not set → deterministic responses for testing

### Safety Limits
- `max_iterations = 10` prevents infinite loops
- Tool execution wrapped in try/except
- JSON parsing with fallback extraction

---

## Future Enhancements

1. **Parallel tool execution** - Call multiple tools simultaneously
2. **Tool dependencies** - Define which tools need others first
3. **Budget constraints** - Limit LLM token usage
4. **Dynamic tool discovery** - Let agents register new tools at runtime
5. **Tool composition** - Create complex workflows from simple tools
6. **Confidence thresholding** - Different strategies for low-confidence skeptic findings
7. **A/B testing** - Compare agentic vs. fixed workflows

---

## Migration Guide (If upgrading existing code)

### For using the orchestrator:
```python
# Same interface - no breaking changes!
orchestrator = AgenticOrchestrator(index=index, memory=memory)
result = orchestrator.run(question=question, objective=objective)
```

### New fields in result:
```python
result.tool_calls  # List of ToolCall objects showing LLM's decisions
result.critique    # Now StructuredCritique instead of just str
```

### Accessing structured critique:
```python
if isinstance(result.critique, StructuredCritique):
    for finding in result.critique.hallucinations:
        print(f"[{finding.severity}] {finding.description}")
else:
    print(result.critique)  # Fallback to string
```

---

## Testing

Run syntax validation:
```bash
python -m py_compile src/agentic_studio/core/tools.py
python -m py_compile src/agentic_studio/core/agent_tools.py
python -m py_compile src/agentic_studio/agents/orchestrator.py
```

Run the app:
```bash
streamlit run streamlit_app.py
```

The system will now:
1. Make reasoned tool-call decisions
2. Produce structured skeptic output
3. Track all tool invocations
4. Be more efficient and adaptive
