from __future__ import annotations

import json
from typing import Any

from agentic_studio.agents.analyst import AnalystAgent
from agentic_studio.agents.planner import PlannerAgent
from agentic_studio.agents.retriever import RetrieverAgent
from agentic_studio.agents.skeptic import SkepticAgent
from agentic_studio.agents.synthesizer import SynthesizerAgent
from agentic_studio.core.agent_tools import set_agents, get_tool_registry, get_last_retrieved_hits
from agentic_studio.core.config import settings
from agentic_studio.core.guardrails import (
    detect_prompt_injection,
    sanitize_user_text,
    trusted_context_prefix,
)
from agentic_studio.core.llm import LLM
from agentic_studio.core.memory import WorkingMemory
from agentic_studio.core.models import (
    AgentResult,
    OrchestrationOutput,
    StructuredCritique,
    ToolCall,
)
from agentic_studio.evals.metrics import evaluate_output
from agentic_studio.rag.index import HybridRAGIndex


class AgenticOrchestrator:
    """Orchestrator that reasons about which tools to call."""

    def __init__(self, index: HybridRAGIndex, memory: WorkingMemory | None = None) -> None:
        llm = LLM()
        self.llm = llm
        self.memory = memory or WorkingMemory()

        # Initialize agents
        self.planner = PlannerAgent(llm)
        self.retriever = RetrieverAgent(index)
        self.analyst = AnalystAgent(llm)
        self.skeptic = SkepticAgent(llm)
        self.synthesizer = SynthesizerAgent(llm)

        # Register agents for tool use
        set_agents(self.planner, self.retriever, self.analyst, self.skeptic, self.synthesizer)

        # Tool tracking
        self.tool_registry = get_tool_registry()
        self.tool_calls: list[ToolCall] = []
        self.max_iterations = settings.max_iterations
        self.iteration = 0

    def run(self, question: str, objective: str) -> OrchestrationOutput:
        """Run agentic orchestration - LLM decides which tools to call."""
        safe_question = sanitize_user_text(question)
        if detect_prompt_injection(safe_question):
            safe_question = (
                "Potential prompt injection detected and blocked. "
                "Please rephrase your request as a domain question only."
            )

        hardened_objective = f"{trusted_context_prefix()} Objective: {objective}"

        # Initialize tracking
        self.tool_calls = []
        self.iteration = 0
        draft_answer = ""
        structured_critique: StructuredCritique | None = None
        final_answer = ""
        plan = ""
        retrieved_docs = []

        # Agentic loop - LLM decides what to do next
        while self.iteration < self.max_iterations:
            self.iteration += 1
            
            print(f"\n{'='*60}")
            print(f"ITERATION {self.iteration} - State Check")
            print(f"{'='*60}")
            print(f"  has_plan: {bool(plan)}")
            print(f"  has_retrieved: {len(retrieved_docs) > 0}")
            print(f"  has_draft: {bool(draft_answer)}")
            print(f"  has_critique: {structured_critique is not None}")
            print(f"  has_final: {bool(final_answer)}")

            # Ask LLM what to do next based on current state
            next_steps = self._decide_next_tools(
                question=safe_question,
                objective=hardened_objective,
                iteration=self.iteration,
                state={
                    "has_plan": bool(plan),
                    "has_retrieved": len(retrieved_docs) > 0,
                    "has_draft": bool(draft_answer),
                    "has_critique": structured_critique is not None,
                    "has_final": bool(final_answer),
                },
            )

            # If LLM says to stop, exit
            if next_steps == "stop":
                print(f"→ LLM decided: STOP")
                break

            if not next_steps:
                print(f"→ No next steps, breaking loop")
                break

            # Execute the decided tools
            for tool_name, args in next_steps:
                print(f"→ Executing tool: {tool_name}")
                
                # Fill in actual values for tools that need them
                if tool_name == "analyze_with_evidence":
                    args["question"] = safe_question
                    args["plan"] = plan
                elif tool_name == "critique_draft":
                    args["draft"] = draft_answer
                    args["question"] = safe_question
                elif tool_name == "synthesize_final_answer":
                    args["draft"] = draft_answer
                    if structured_critique:
                        args["critique"] = structured_critique.to_text()

                tool_call = ToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    reasoning=f"Iteration {self.iteration}: LLM decided to call {tool_name}",
                )

                result = self._execute_tool(tool_call, safe_question, plan)
                self.tool_calls.append(tool_call)
                self.memory.add(tool_name, str(result)[:1000])
                
                print(f"  → Result: {str(result)[:100]}...")

                # Store results based on tool
                if tool_name == "create_analysis_plan":
                    plan = str(result)
                    print(f"  ✓ Plan created (length: {len(plan)})")
                elif tool_name == "retrieve_evidence":
                    retrieved_docs = get_last_retrieved_hits()
                    print(f"  ✓ Retrieved {len(retrieved_docs)} documents")
                elif tool_name == "analyze_with_evidence":
                    draft_answer = str(result)
                    print(f"  ✓ Draft analysis created (length: {len(draft_answer)})")
                elif tool_name == "critique_draft":
                    if isinstance(result, StructuredCritique):
                        structured_critique = result
                        print(f"  ✓ Structured critique created")
                        print(f"    - Hallucinations: {len(structured_critique.hallucinations)}")
                        print(f"    - Weak points: {len(structured_critique.weak_points)}")
                        print(f"    - Missing evidence: {len(structured_critique.missing_evidence)}")
                elif tool_name == "synthesize_final_answer":
                    final_answer = str(result)
                    print(f"  ✓ Final answer synthesized (length: {len(final_answer)})")

            # Check if we're done
            if final_answer and structured_critique:
                print(f"\n→ All components ready. Breaking loop.")
                break

        # Fallback - if something missing, do basic generation (COMMENTED OUT - PURE AGENTIC MODE)
        # if not plan:
        #     plan = self.planner.run(safe_question, hardened_objective)
        # if not retrieved_docs:
        #     retrieved_docs = self.retriever.run(safe_question, top_k=settings.max_chunks)
        # if not draft_answer:
        #     draft_answer = self.analyst.run(safe_question, plan, retrieved_docs, self.memory.render())
        # if not structured_critique:
        #     structured_critique = self.skeptic.run(draft_answer, safe_question)
        # if not final_answer:
        #     critique_text = structured_critique.to_text() if isinstance(structured_critique, StructuredCritique) else str(structured_critique)
        #     final_answer = self.synthesizer.run(draft_answer, critique_text)

        # Prepare trace
        trace = [
            AgentResult(
                name="orchestrator",
                output=f"Used agentic reasoning over {self.iteration} iterations",
                reasoning="LLM-driven tool selection based on state",
            ),
        ]

        # Evaluate
        metrics = evaluate_output(final_answer, retrieved_docs)

        return OrchestrationOutput(
            final_answer=final_answer,
            citations=retrieved_docs,
            critique=structured_critique.to_text() if isinstance(structured_critique, StructuredCritique) else (structured_critique or "No critique generated"),
            plan=plan,
            metrics=metrics,
            trace=trace,
            tool_calls=self.tool_calls,
        )

    def _decide_next_tools(
        self,
        question: str,
        objective: str,
        iteration: int,
        state: dict[str, bool],
    ) -> list[tuple[str, dict]] | str:
        """LLM decides what to do next based on current state."""
        
        state_str = "\n".join([f"  • {k}: {v}" for k, v in state.items()])
        
        prompt = (
            f"{objective}\n\n"
            f"Current Progress (iteration {iteration}):\n{state_str}\n\n"
            f"Question: {question[:100]}...\n\n"
            "YOU MUST FOLLOW THIS DECISION TREE EXACTLY:\n\n"
            "1. has_plan == False? → ANSWER: create_analysis_plan\n"
            "2. has_retrieved == False? → ANSWER: retrieve_evidence\n"
            "3. has_draft == False? → ANSWER: analyze_with_evidence\n"
            "4. has_critique == False? → ANSWER: critique_draft\n"
            "5. has_final == False? → ANSWER: synthesize_final_answer\n"
            "6. All True? → ANSWER: stop\n\n"
            "DO NOT DEVIATE FROM THIS TREE.\n"
            "Current state check:\n"
            f"- has_plan: {state.get('has_plan')}\n"
            f"- has_retrieved: {state.get('has_retrieved')}\n"
            f"- has_draft: {state.get('has_draft')}\n"
            f"- has_critique: {state.get('has_critique')}\n"
            f"- has_final: {state.get('has_final')}\n\n"
            "Respond ONLY with the action name: create_analysis_plan, retrieve_evidence, analyze_with_evidence, critique_draft, synthesize_final_answer, or stop"
        )

        print(f"\n--- DECISION PROMPT (Iteration {iteration}) ---")
        print(f"State: {state}")
        print(f"Full Prompt:\n{prompt}")
        print(f"--- END PROMPT ---\n")

        response = self.llm.generate(prompt)
        response_lower = response.text.lower().strip()
        
        print(f"LLM Raw Response: '{response.text}'")
        print(f"LLM Response (lowercase/stripped): '{response_lower}'\n")

        # Extract the action keyword - look for exact matches
        actions = [
            "create_analysis_plan",
            "retrieve_evidence",
            "analyze_with_evidence",
            "critique_draft",
            "synthesize_final_answer",
            "stop",
        ]

        next_action = None
        for action in actions:
            if action in response_lower:
                next_action = action
                print(f"✓ Parsed action: {next_action}")
                break

        # Compute the mandatory next action according to state machine
        mandatory_action = None
        if not state.get("has_plan"):
            mandatory_action = "create_analysis_plan"
        elif not state.get("has_retrieved"):
            mandatory_action = "retrieve_evidence"
        elif not state.get("has_draft"):
            mandatory_action = "analyze_with_evidence"
        elif not state.get("has_critique"):
            mandatory_action = "critique_draft"
        elif not state.get("has_final"):
            mandatory_action = "synthesize_final_answer"
        else:
            mandatory_action = "stop"

        if not next_action:
            print(f"✗ Could not parse action from response, using state machine:")
            next_action = mandatory_action
            print(f"  → Mandatory action: {next_action}")
        elif next_action != mandatory_action and next_action != "stop":
            # LLM chose wrong action - override with state machine
            print(f"⚠ LLM chose '{next_action}' but state requires '{mandatory_action}'")
            print(f"   [ENFORCING STATE MACHINE]")
            next_action = mandatory_action

        if next_action == "stop":
            return "stop"

        # Convert action to tool call
        tools_to_call: list[tuple[str, dict]] = [(next_action, {})]
        return tools_to_call

    def _execute_tool(
        self,
        tool_call: ToolCall,
        question: str,
        plan: str,
    ) -> str | Any:
        """Execute a tool call and return the result."""
        try:
            tool_name = tool_call.tool_name
            args = tool_call.arguments

            # Provide default values if arguments are missing
            if tool_name == "create_analysis_plan":
                return self.tool_registry.execute(
                    tool_name,
                    question=args.get("question", question),
                    objective=args.get("objective", ""),
                )
            elif tool_name == "retrieve_evidence":
                return self.tool_registry.execute(
                    tool_name,
                    query=args.get("query", question),
                    top_k=args.get("top_k", settings.max_chunks),
                )
            elif tool_name == "analyze_with_evidence":
                return self.tool_registry.execute(
                    tool_name,
                    question=args.get("question", question),
                    plan=args.get("plan", plan),
                    memory_context=self.memory.render(),
                )
            elif tool_name == "critique_draft":
                # This returns a StructuredCritique object
                return self.skeptic.run(
                    draft=args.get("draft", ""),
                    question=args.get("question", question),
                )
            elif tool_name == "synthesize_final_answer":
                return self.tool_registry.execute(
                    tool_name,
                    draft=args.get("draft", ""),
                    critique=args.get("critique", ""),
                )
            else:
                return f"[ERROR] Unknown tool: {tool_name}"
        except Exception as e:
            return f"[ERROR] Tool execution failed: {str(e)}"