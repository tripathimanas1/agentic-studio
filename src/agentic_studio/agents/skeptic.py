from __future__ import annotations

import json

from agentic_studio.core.llm import LLM
from agentic_studio.core.models import SkepticFinding, StructuredCritique


class SkepticAgent:
    """Agent that critically analyzes answers and produces structured findings."""

    name = "skeptic"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def run(self, draft: str, question: str) -> StructuredCritique:
        """Analyze draft and return structured critique."""
        # First, get the structured analysis
        structured_critique = self._analyze_structured(draft, question)
        return structured_critique

    def _analyze_structured(self, draft: str, question: str) -> StructuredCritique:
        """Use LLM-driven structured prompting to get categorized findings."""
        prompt = (
            "You are a critical analyst specializing in identifying flaws and risks in strategic proposals. "
            "Analyze the draft answer and identify credible issues across these categories:\n\n"
            "1. HALLUCINATIONS: Claims that are unsupported by evidence or contradict stated information\n"
            "2. ATTACK_VECTORS: Security, financial, or execution risks that could be exploited or cause failure\n"
            "3. WEAK_POINTS: Assumptions, logical leaps, or arguments that lack sufficient support\n"
            "4. MISSING_EVIDENCE: Critical information, alternatives, or counterarguments not addressed\n"
            "5. OVERCONFIDENCE: Claims that overstate certainty or success probability without caveats\n"
            "6. LOGICAL_FLAWS: Reasoning errors, unvalidated dependencies, or false causality\n\n"
            "Return VALID JSON (all fields required):\n"
            "{\n"
            '  "hallucinations": [{"severity": "critical|high|medium|low", "description": "...", "evidence": "..."}],\n'
            '  "attack_vectors": [{"severity": "critical|high|medium|low", "description": "...", "recommendation": "..."}],\n'
            '  "weak_points": [{"severity": "critical|high|medium|low", "description": "...", "evidence": "..."}],\n'
            '  "missing_evidence": [{"severity": "critical|high|medium|low", "description": "..."}],\n'
            '  "overconfidence": [{"severity": "critical|high|medium|low", "description": "..."}],\n'
            '  "logical_flaws": [{"severity": "critical|high|medium|low", "description": "..."}],\n'
            '  "overall_confidence": 0.65,\n'
            '  "summary": "Concise summary of major issues found"\n'
            "}\n\n"
            "You may have 0 or more items in each category. Return ONLY valid JSON, no markdown or extra text.\n\n"
            f"Question:\n{question}\n\n"
            f"Draft Answer:\n{draft}"
        )

        # Try structured generation first
        response_dict = self.llm.generate_structured(prompt, response_schema={})

        # Parse the response
        if isinstance(response_dict, dict) and not response_dict.get("mock"):
            try:
                return self._parse_critique_response(response_dict)
            except Exception as e:
                pass

        # Fallback: use regular generation and parse
        response = self.llm.generate(prompt)
        try:
            json_str = self._extract_json(response.text)
            if json_str and json_str != "{}":
                response_dict = json.loads(json_str)
                return self._parse_critique_response(response_dict)
        except Exception:
            pass
        
        # Final fallback: return empty but valid critique
        return StructuredCritique(
            summary="Unable to generate structured critique at this time.",
            overall_confidence=0.3,
            hallucinations=[],
            attack_vectors=[],
            weak_points=[],
            missing_evidence=[],
            overconfidence=[],
            logical_flaws=[],
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain markdown or other content."""
        # Try to find JSON object
        start_idx = text.find("{")
        if start_idx == -1:
            return "{}"

        # Find matching closing brace
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

        return text[start_idx:]

    def _parse_critique_response(self, response_dict: dict) -> StructuredCritique:
        """Parse API response into StructuredCritique."""

        def parse_findings(findings_data: list | None, category: str) -> list[SkepticFinding]:
            if not findings_data or not isinstance(findings_data, list):
                return []

            findings = []
            for item in findings_data:
                if not isinstance(item, dict):
                    continue

                findings.append(
                    SkepticFinding(
                        category=category,
                        severity=item.get("severity", "medium"),
                        description=item.get("description", ""),
                        evidence=item.get("evidence", ""),
                        recommendation=item.get("recommendation", ""),
                    )
                )
            return findings

        return StructuredCritique(
            hallucinations=parse_findings(response_dict.get("hallucinations"), "hallucination"),
            attack_vectors=parse_findings(response_dict.get("attack_vectors"), "attack"),
            weak_points=parse_findings(response_dict.get("weak_points"), "weak_point"),
            missing_evidence=parse_findings(response_dict.get("missing_evidence"), "missing_evidence"),
            overconfidence=parse_findings(response_dict.get("overconfidence"), "overconfidence"),
            logical_flaws=parse_findings(response_dict.get("logical_flaws"), "logical_flaw"),
            overall_confidence=float(response_dict.get("overall_confidence", 0.5)),
            summary=str(response_dict.get("summary", "")),
        )

    def _create_structured_findings(self, critique_text: str, draft: str, question: str) -> StructuredCritique:
        """Return empty critique when LLM parsing fails. No hardcoded logic."""
        return StructuredCritique(
            summary="Unable to parse LLM response. Please retry.",
            overall_confidence=0.0,
            hallucinations=[],
            attack_vectors=[],
            weak_points=[],
            missing_evidence=[],
            overconfidence=[],
            logical_flaws=[],
        )