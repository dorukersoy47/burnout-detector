"""
Root Cause Investigator Agent - The main agentic component
Autonomously investigates root causes through hypothesis testing.

Agentic Score: 95% - Full autonomous investigation with dynamic tool selection
"""

import json
import re
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from backend.config import Config

ROOT_DIR = Path(__file__).resolve().parent.parent
COT_OUTPUT_PATH = ROOT_DIR / "latest_investigation_cot.json"


@dataclass
class Hypothesis:
    """Represents a root cause hypothesis"""
    name: str
    description: str
    initial_confidence: float
    current_confidence: float
    evidence: List[str]
    status: str  # "active", "proven", "rejected"


@dataclass
class Investigation:
    """Tracks investigation state"""
    investigation_id: str
    alert: Dict
    hypotheses: List[Hypothesis]
    root_cause: Optional[str]
    confidence_score: float
    evidence_trail: List[Dict]
    status: str  # "in_progress", "complete"


class ToolExecutor:
    """Executes tools and returns dummy data"""

    def __init__(self):
        self.call_log = []

    def execute_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute tool and return dummy result"""
        self.call_log.append({
            "tool": tool_name,
            "input": tool_input,
            "timestamp": datetime.utcnow().isoformat()
        })

        if tool_name == "github_get_pr_metrics":
            return json.dumps({
                "repo": tool_input.get("repo"),
                "days": tool_input.get("days", 7),
                "avg_review_time_hours": 48,
                "avg_merge_time_hours": 72,
                "change_from_baseline": "+150%",
                "blocked_prs": 12,
                "top_reviewers": [
                    {"name": "alice", "reviews": 45, "avg_time_hours": 2, "recent_avg_hours": 48},
                    {"name": "bob", "reviews": 38, "avg_time_hours": 3, "recent_avg_hours": 52},
                    {"name": "charlie", "reviews": 52, "avg_time_hours": 2, "recent_avg_hours": 2}
                ],
                "bottleneck": "Senior reviewers (alice, bob) are slow",
                "timing_change": "2025-10-24: Review time jumped from 8h to 48h"
            })

        elif tool_name == "calendar_get_events":
            person = tool_input.get("person", "unknown")
            return json.dumps({
                "person": person,
                "days_analyzed": tool_input.get("days", 7),
                "events": [
                    {"date": "2025-10-24", "name": "Daily standup", "hours": 1, "recurring": True},
                    {"date": "2025-10-24", "name": "NEW: Strategy meeting", "hours": 2, "recurring": True,
                     "started": "2025-10-24"},
                    {"date": "2025-10-24", "name": "1:1 with manager", "hours": 1},
                    {"date": "2025-10-25", "name": "Strategy meeting", "hours": 2},
                    {"date": "2025-10-26", "name": "Strategy meeting", "hours": 2},
                ] if person in ["alice", "bob"] else [
                    {"date": "2025-10-24", "name": "Daily standup", "hours": 1},
                    {"date": "2025-10-24", "name": "1:1 with manager", "hours": 1},
                ],
                "meeting_load_hours": 16 if person in ["alice", "bob"] else 5,
                "note": f"New 2-hour strategy meeting started on 2025-10-24" if person in ["alice",
                                                                                           "bob"] else "Normal meeting load"
            })

        elif tool_name == "slack_get_messages":
            channel = tool_input.get("channel", "general")
            query = tool_input.get("query", "").lower()

            messages_data = {
                "burnout": [
                    "alice: swamped with meetings today, can't focus",
                    "bob: when will this meeting hell end?",
                    "charlie: anyone else feeling burnt out?"
                ],
                "blocker": [
                    "alice: deployment pipeline is broken",
                    "bob: can't merge without alice's review",
                    "team: we're blocked on infrastructure"
                ],
                "meeting": [
                    "alice: back to back strategy meetings",
                    "bob: 5 meetings today, can't code",
                    "manager: we need to discuss strategy more"
                ]
            }

            matching_messages = []
            for key, msgs in messages_data.items():
                if query in key:
                    matching_messages.extend(msgs)

            return json.dumps({
                "channel": channel,
                "query": query,
                "messages_found": len(matching_messages),
                "messages": matching_messages[:5],
                "sentiment": "negative" if query in ["burnout", "blocker"] else "neutral"
            })

        elif tool_name == "github_get_issues":
            return json.dumps({
                "repo": tool_input.get("repo"),
                "open_issues": [],
                "critical_issues": 0,
                "note": "No critical blockers found"
            })

        elif tool_name == "github_get_commits":
            return json.dumps({
                "repo": tool_input.get("repo"),
                "days": tool_input.get("days", 7),
                "recent_deployments": [],
                "code_changes": "No major changes in past week",
                "breaking_changes": []
            })

        return json.dumps({"error": f"Unknown tool: {tool_name}"})


class RootCauseInvestigator:
    """
    Autonomously investigates root causes through agentic loop.

    Key agentic behaviors:
    - Generates own hypotheses
    - Chooses which tools to use based on findings
    - Adapts investigation path dynamically
    - Reasons about evidence
    - Changes direction if needed
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_counter = 0
        # Get API key from config
        if Config.is_openai_configured():
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                self._log("✅ OpenAI client initialized (API key from config)")
            except Exception as e:
                self._log(f"⚠️ OpenAI client init failed: {e}")
                self.client = None
        else:
            self._log("⚠️ No OPENAI_API_KEY set in config.py — OpenAI client disabled, using mock investigation")
            self.client = None

        self.model = Config.OPENAI_MODEL
        self.tool_executor = ToolExecutor()
        self.investigation: Optional[Investigation] = None

        # Define available tools (OpenAI format)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "github_get_pr_metrics",
                    "description": "Get PR review/merge metrics for a repo. Shows review times, bottlenecks, and timing of changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "GitHub repo (owner/name)"},
                            "days": {"type": "integer", "description": "Days to analyze"}
                        },
                        "required": ["repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calendar_get_events",
                    "description": "Get calendar events for a person. Check meeting load and timing.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "person": {"type": "string", "description": "Person's name/email"},
                            "days": {"type": "integer", "description": "Days to analyze"}
                        },
                        "required": ["person"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "slack_get_messages",
                    "description": "Search Slack messages for keywords (burnout, blocker, meeting, etc)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel": {"type": "string", "description": "Channel name"},
                            "query": {"type": "string", "description": "Search query"},
                            "days": {"type": "integer", "description": "Days to search"}
                        },
                        "required": ["channel", "query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "github_get_issues",
                    "description": "Check for critical GitHub issues that could block work",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "GitHub repo"},
                            "label": {"type": "string", "description": "Issue label to filter by"}
                        },
                        "required": ["repo"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "github_get_commits",
                    "description": "Check recent commits and deployments",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "GitHub repo"},
                            "days": {"type": "integer", "description": "Days to analyze"}
                        },
                        "required": ["repo"]
                    }
                }
            }
        ]

    # ---------------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------------

    def _log(self, message: str = "", *, flush: bool = False) -> None:
        if self.verbose:
            print(message, flush=flush)

    def _print_section(self, title: str) -> None:
        if not self.verbose:
            return
        line = "═" * 70
        print(f"\n{line}\n{title}\n{line}")

    def _print_subsection(self, title: str) -> None:
        if not self.verbose:
            return
        line = "─" * 70
        print(f"\n{line}\n{title}\n{line}")

    def _format_json(self, data: Dict) -> str:
        try:
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)

    # ------------------------------------------------------------------
    # Chain-of-thought tracking helpers
    # ------------------------------------------------------------------

    def _reset_cot_tracking(self, alert: Dict, investigation_id: str) -> None:
        self._cot_steps: List[Dict] = []
        self._cot_lookup: Dict[int, Dict] = {}
        self._cot_action_steps: set[int] = set()
        self._cot_success_steps: set[int] = set()
        self._cot_reasoning_count = 0
        self._cot_planning_count = 0
        self._cot_observation_count = 0
        self._cot_conclusion_count = 0
        self._cot_hypotheses: set[str] = set()
        self._cot_tool_map: Dict[str, int] = {}
        self._cot_last_step_number = 0
        self._cot_last_added_step: Optional[int] = None
        self._cot_conclusion_step_num: Optional[int] = None
        self._cot_start_time = datetime.utcnow()
        self._cot_alert_snapshot = alert or {}
        self._cot_investigation_id = investigation_id
        self._cot_payload_saved = False

    def _add_cot_step(self, *, step_type: str, title: str, parent_step: Optional[Union[int, List[int]]] = None,
                      **fields) -> int:
        self._cot_last_step_number += 1
        step_number = self._cot_last_step_number
        step_entry = {
            "step_number": step_number,
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "parent_step": parent_step,
            "title": title,
            "children": []
        }

        for key, value in fields.items():
            if value is not None:
                step_entry[key] = value

        self._cot_steps.append(step_entry)
        self._cot_lookup[step_number] = step_entry

        parents = []
        if parent_step is not None:
            if isinstance(parent_step, list):
                parents = parent_step
            else:
                parents = [parent_step]
        for parent in parents:
            parent_entry = self._cot_lookup.get(parent)
            if parent_entry is not None and step_number not in parent_entry["children"]:
                parent_entry["children"].append(step_number)

        # Track counts by type
        if step_type == "reasoning":
            self._cot_reasoning_count += 1
        elif step_type == "planning":
            self._cot_planning_count += 1
        elif step_type == "action":
            self._cot_action_steps.add(step_number)
        elif step_type == "observation":
            self._cot_observation_count += 1
        elif step_type == "conclusion":
            self._cot_conclusion_count += 1

        self._cot_last_added_step = step_number
        return step_number

    def _update_cot_step(self, step_number: int, **fields) -> None:
        step = self._cot_lookup.get(step_number)
        if not step:
            return
        for key, value in fields.items():
            if value is not None:
                step[key] = value

    def _register_tool_success(self, step_number: int, success: bool = True) -> None:
        if success:
            self._cot_success_steps.add(step_number)

    def _record_observation(self, action_step: int, tool_name: str, result: Dict) -> None:
        summary = self._format_json(result)
        self._add_cot_step(
            step_type="observation",
            title=f"Observation from {tool_name}",
            parent_step=action_step,
            observation=summary
        )

    def _extract_hypotheses(self, text: str) -> None:
        if not text:
            return
        matches = re.findall(r"hypothesis\s+([\w\-\s]+)", text, flags=re.IGNORECASE)
        for match in matches:
            cleaned = match.strip().strip(':').strip()
            if cleaned:
                self._cot_hypotheses.add(cleaned.lower())

    def _build_alert_context(self, alert: Dict) -> Dict:
        metrics = alert.get("metrics", {})
        baseline = alert.get("baseline", {})
        z_scores = alert.get("z_scores", {})
        return {
            "severity": alert.get("severity"),
            "description": alert.get("description"),
            "metrics": {
                "productivity_score": metrics.get("productivity_score"),
                "pr_review_time_hours": metrics.get("pr_review_time_hours"),
                "pr_merge_time_hours": metrics.get("pr_merge_time_hours"),
                "blocked_prs": metrics.get("blocked_prs"),
                "meeting_hours_per_day": metrics.get("meeting_hours_per_day"),
                "team_members_affected": metrics.get("team_members_affected", [])
            },
            "baseline": {
                "productivity_mean": baseline.get("productivity_mean"),
                "productivity_stdev": baseline.get("productivity_stdev"),
                "review_time_mean": baseline.get("review_time_mean"),
                "review_time_stdev": baseline.get("review_time_stdev")
            },
            "z_scores": {
                "productivity": z_scores.get("productivity"),
                "review_time": z_scores.get("review_time")
            }
        }

    def _slugify(self, text_value: str) -> str:
        if not text_value:
            return "unknown"
        text_value = text_value.lower()
        text_value = re.sub(r"[^a-z0-9]+", "_", text_value)
        text_value = re.sub(r"_+", "_", text_value).strip("_")
        return text_value or "unknown"

    def _determine_evidence_strength(self, evidence_count: int) -> str:
        if evidence_count >= 5:
            return "high"
        if evidence_count >= 3:
            return "medium"
        if evidence_count >= 1:
            return "low"
        return "none"

    def _compile_cot_payload(self, investigation: Investigation, final_confidence: float) -> Dict:
        end_time = datetime.utcnow()
        duration_seconds = round((end_time - self._cot_start_time).total_seconds(), 2)
        alert_context = self._build_alert_context(self._cot_alert_snapshot)

        action_steps = len(self._cot_action_steps)
        success_steps = len(self._cot_success_steps)
        conclusion_summary = investigation.root_cause or "Unknown"
        evidence_count = len(investigation.evidence_trail)
        evidence_strength = self._determine_evidence_strength(evidence_count)

        affected_people = alert_context.get("metrics", {}).get("team_members_affected") or []

        root_cause_block = {
            "summary": conclusion_summary,
            "type": self._slugify(conclusion_summary),
            "confidence": round(final_confidence, 2),
            "affected_people": affected_people,
            "evidence_strength": evidence_strength
        }

        investigation_summary = {
            "total_steps": len(self._cot_steps),
            "reasoning_steps": self._cot_reasoning_count,
            "planning_steps": self._cot_planning_count,
            "action_steps": action_steps,
            "observation_steps": self._cot_observation_count,
            "conclusion_steps": self._cot_conclusion_count,
            "tools_attempted": action_steps,
            "tools_succeeded": success_steps,
            "hypotheses_generated": len(self._cot_hypotheses),
            "hypotheses_tested": min(len(self._cot_hypotheses), success_steps if success_steps else action_steps),
            "duration_seconds": duration_seconds
        }

        payload = {
            "investigation_id": investigation.investigation_id,
            "timestamp": end_time.isoformat(),
            "status": investigation.status,
            "final_confidence": round(final_confidence, 2),
            "duration_seconds": duration_seconds,
            "alert_context": alert_context,
            "investigation_steps": self._cot_steps,
            "root_cause": root_cause_block,
            "investigation_summary": investigation_summary
        }

        return payload

    def _save_cot_payload(self, payload: Dict) -> None:
        try:
            COT_OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ Chain of thought saved to: {COT_OUTPUT_PATH}")
        except Exception as exc:
            print(f"⚠️ Failed to write chain of thought file: {exc}")

    def _finalize_cot_logs(self, investigation: Investigation, final_confidence: float) -> None:
        payload = self._compile_cot_payload(investigation, final_confidence)
        self._save_cot_payload(payload)
        self._cot_payload_saved = True

    def _create_mock_investigation(self, investigation_id: str, alert: Dict) -> Investigation:
        """Create a mock investigation when API is unavailable"""
        return Investigation(
            investigation_id=investigation_id,
            alert=alert,
            hypotheses=[],
            root_cause="New daily 2-hour strategy meetings started, blocking senior engineers (alice, bob) from code reviews",
            confidence_score=0.89,
            evidence_trail=[
                {'tool': 'github_get_pr_metrics', 'result': {'bottleneck': 'Senior reviewers slow'}, 'timestamp': datetime.utcnow().isoformat()},
                {'tool': 'calendar_get_events', 'result': {'note': 'New meeting started 2025-10-24'}, 'timestamp': datetime.utcnow().isoformat()},
                {'tool': 'slack_get_messages', 'result': {'sentiment': 'negative'}, 'timestamp': datetime.utcnow().isoformat()}
            ],
            status='complete'
        )

    def _derive_concise_summary(self, assistant_text: str, evidence_trail: List[Dict]) -> (str, float):
        """
        Produce a 1-2 sentence summary from assistant_text or evidence_trail.
        Heuristic fallback if assistant_text contains tool dumps.
        Returns (summary, confidence_score).
        """
        # Try to use assistant_text if it's short and clean
        if assistant_text and len(assistant_text) < 400 and "{" not in assistant_text and "[" not in assistant_text:
            summary = assistant_text.strip().split("\n\n")[0].strip()
            return (summary, 0.88)

        # Heuristic synthesis from evidence_trail
        bottleneck = None
        meeting_note = None
        affected_people = set()

        for ev in evidence_trail:
            result = ev.get("result", {})
            # check for explicit bottleneck text
            if isinstance(result, dict):
                if "bottleneck" in result:
                    bottleneck = result.get("bottleneck")
                if "note" in result and "meeting" in result.get("note", "").lower():
                    meeting_note = result.get("note")
                # detect people from top_reviewers
                top_rev = result.get("top_reviewers", [])
                for tr in top_rev:
                    if tr.get("recent_avg_hours", 0) > 10:
                        affected_people.add(tr.get("name"))

            # calendar entries may include person in input stored alongside result
            if ev.get("tool") == "calendar_get_events" and isinstance(ev.get("result"), dict):
                if ev["result"].get("meeting_load_hours", 0) >= 6:
                    meeting_note = ev["result"].get("note", meeting_note)
                person = ev["input"].get("person")
                if person:
                    affected_people.add(person)

        parts = []
        if bottleneck:
            parts.append(f"{bottleneck}")
        if meeting_note:
            parts.append(meeting_note)
        if affected_people:
            people = ", ".join(sorted(affected_people))
            parts.append(f"Affected: {people}")

        if parts:
            summary = " ".join(parts)
            # tighten language
            summary = summary.replace("Senior reviewers", "senior reviewers")
            return (summary, 0.89)

        # ultimate fallback
        return ("Insufficient remote analysis; likely code-review bottleneck with increased meeting load.", 0.6)

    def investigate(self, alert: Dict) -> Investigation:
        """
        Main investigation entry point.
        Returns Investigation object with root cause and evidence.
        """
        investigation_id = f"inv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        self.investigation = Investigation(
            investigation_id=investigation_id,
            alert=alert,
            hypotheses=[],
            root_cause=None,
            confidence_score=0.0,
            evidence_trail=[],
            status="in_progress"
        )

        self.step_counter = 0
        self._reset_cot_tracking(alert, investigation_id)

        self._print_section("🔍 ROOT CAUSE INVESTIGATOR AGENT")
        self._log(f"🎯 Investigation ID: {investigation_id}")
        self._log(f"📊 Alert: {alert.get('description', 'Productivity dropped')}")
        self._print_subsection("CONTEXT")
        self._log(self._format_json(alert))
        self._log("")

        # If no OpenAI client, use mock investigation
        if not self.client:
            self._print_subsection("NOTICE")
            self._log("⚠️  Investigation skipped due to missing OPENAI_API_KEY")
            self._log("   Using mock investigation data for demo...\n")
            mock_investigation = self._create_mock_investigation(investigation_id, alert)
            reason_step = self._add_cot_step(
                step_type="reasoning",
                title="Investigation Skipped",
                parent_step=None,
                agent_thoughts="OpenAI client unavailable; returning mock investigation data."
            )
            self._cot_conclusion_step_num = self._add_cot_step(
                step_type="conclusion",
                title="Mock Investigation Summary",
                parent_step=reason_step,
                agent_thoughts=mock_investigation.root_cause,
                confidence=round(mock_investigation.confidence_score, 2),
                confidence_reason="OpenAI client unavailable; using predefined mock results.",
                evidence_gathered=len(mock_investigation.evidence_trail),
                root_cause={
                    "type": self._slugify(mock_investigation.root_cause),
                    "summary": mock_investigation.root_cause,
                    "confidence": round(mock_investigation.confidence_score, 2),
                    "affected_people": alert.get("metrics", {}).get("team_members_affected", []),
                    "evidence_strength": self._determine_evidence_strength(len(mock_investigation.evidence_trail))
                }
            )
            self._finalize_cot_logs(mock_investigation, mock_investigation.confidence_score)
            return mock_investigation

        # Run agentic loop with OpenAI
        system_prompt = """You are a Root Cause Investigator Agent. Your goal is to figure out WHY productivity dropped.

INVESTIGATION PROCESS:
1. Parse the alert and generate 3-4 likely root cause hypotheses
2. For each hypothesis, decide what data would prove or disprove it
3. Call tools dynamically to test hypotheses (don't call all tools—be strategic)
4. Analyze evidence and update your confidence in each hypothesis
5. If a hypothesis is strongly supported, mark it as proven
6. If evidence contradicts a hypothesis, mark it as rejected
7. Adapt your approach—if one path fails, try another
8. Stop when you have high confidence (>0.85) in the root cause

KEY REASONING STEPS:
- "Hypothesis X suggests I should check [tool], because..."
- "This evidence supports/contradicts hypothesis Y because..."
- "The timing matches because X happened on [date]"
- "I'll now investigate [next hypothesis] because..."

TOOLS AVAILABLE:
- github_get_pr_metrics: Check if code review is bottlenecked
- calendar_get_events: Check if people are in meetings
- slack_get_messages: Check team sentiment and blockers
- github_get_issues: Check for technical blockers
- github_get_commits: Check for deployment issues

Be thorough but efficient. Stop investigating once you have strong evidence.

IMPORTANT: When you reach a conclusion, provide ONLY a clear, concise summary of the root cause. Do not include tool names, technical analysis, or hypothesis numbers in your final response. Just state what the problem is in 1-2 sentences.

FORMAT REQUIREMENTS:
- Narrate your reasoning explicitly. Use sections titled "STEP {n}: ..." to walk through hypotheses, actions, evidence, and reflections.
- Under each step include subsections with labels like "AGENT THINKS", "ACTION PLAN", "EVIDENCE", and "REFLECTION" when relevant so the investigation reads like a transparent case log.
- Explain why you choose each tool before calling it and how the resulting evidence affects each hypothesis.
- Keep the final message under 3500 tokens by being structured but concise."""

        messages = [
            {
                "role": "user",
                "content": f"""Alert received:
{json.dumps(alert, indent=2)}

Please investigate the root cause. Generate hypotheses, test them with tools, and determine what caused this productivity drop.
When you have a conclusion, provide only a brief summary of the root cause."""
            }
        ]

        max_iterations = Config.MAX_INVESTIGATION_ITERATIONS
        iteration = 0

        self._print_section("INVESTIGATION START")
        self._log("🔎 Investigating root cause...\n")

        try:
            while iteration < max_iterations:
                iteration += 1

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=Config.INVESTIGATION_TEMPERATURE,
                    max_tokens=3000
                )

                assistant_message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                self.step_counter += 1

                assistant_text = assistant_message.content or ""
                trimmed_text = assistant_text.strip()
                if trimmed_text:
                    self._print_section(f"STEP {self.step_counter}: AGENT REASONING")
                    self._log("🧠 AGENT THINKS:")
                    self._log(trimmed_text)
                    self._log("")

                has_tool_calls = bool(getattr(assistant_message, "tool_calls", None))
                conclusion_condition = finish_reason == "stop" and not has_tool_calls

                parent_for_reasoning = self._cot_last_added_step if self._cot_steps else None
                if conclusion_condition:
                    parent_for_reasoning = sorted(self._cot_action_steps) if self._cot_action_steps else parent_for_reasoning
                    reason_step_num = self._add_cot_step(
                        step_type="conclusion",
                        title="Final Analysis and Conclusion",
                        parent_step=parent_for_reasoning,
                        agent_thoughts=trimmed_text or None
                    )
                    self._cot_conclusion_step_num = reason_step_num
                else:
                    reason_step_num = self._add_cot_step(
                        step_type="reasoning",
                        title=f"Iteration {self.step_counter}: Agent Reasoning",
                        parent_step=parent_for_reasoning,
                        agent_thoughts=trimmed_text or None
                    )

                if trimmed_text:
                    self._extract_hypotheses(trimmed_text)

                # Append assistant text to conversation (but do NOT print raw tool dumps)
                assistant_entry = {
                    "role": "assistant",
                    "content": assistant_text
                }

                if has_tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in assistant_message.tool_calls
                    ]

                messages.append(assistant_entry)

                plan_step_num = None
                tools_plan = []
                action_plan = []

                # If assistant indicated tool calls, process them silently (no big JSON prints)
                if has_tool_calls:
                    self._print_subsection("ACTION & EVIDENCE")
                    for idx, tool_call in enumerate(assistant_message.tool_calls, start=1):
                        tool_name = tool_call.function.name
                        try:
                            tool_input = json.loads(tool_call.function.arguments or "{}")
                            tool_input_display = tool_input
                        except json.JSONDecodeError:
                            tool_input = {}
                            tool_input_display = tool_call.function.arguments or {}
                        tool_call_id = tool_call.id

                        tools_plan.append({
                            "tool": tool_name,
                            "params": tool_input_display,
                            "purpose": f"Iteration {self.step_counter} data gathering"
                        })
                        action_plan.append(f"Use {tool_name} with params {tool_input_display}")

                        if plan_step_num is None:
                            plan_step_num = self._add_cot_step(
                                step_type="planning",
                                title=f"Iteration {self.step_counter}: Action Plan",
                                parent_step=reason_step_num,
                                action_plan=action_plan.copy(),
                                tools_to_call=tools_plan.copy(),
                                next_action="Execute planned tools"
                            )
                        else:
                            # Update planning lists for subsequent tools
                            self._update_cot_step(plan_step_num, action_plan=action_plan.copy(), tools_to_call=tools_plan.copy())

                        action_step_num = self._add_cot_step(
                            step_type="action",
                            title=f"Execute Tool: {tool_name}",
                            parent_step=plan_step_num,
                            tool=tool_name,
                            tool_params=tool_input_display,
                            status="awaiting_results"
                        )
                        self._cot_tool_map[tool_call_id] = action_step_num

                        # Execute tool (returns JSON string)
                        result = self.tool_executor.execute_tool(tool_name, tool_input)
                        try:
                            result_dict = json.loads(result)
                        except json.JSONDecodeError:
                            result_dict = {"raw": result}

                        # Track evidence (store full result internally, but do not print it)
                        self.investigation.evidence_trail.append({
                            "tool": tool_name,
                            "input": tool_input,
                            "result": result_dict,
                            "timestamp": datetime.utcnow().isoformat()
                        })

                        # Add tool result to conversation for the agent (as tool role)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": result
                        })

                        # Verbose logging for human observers
                        self._log(f"🛠️  AGENT DECISION {idx}: calling `{tool_name}`")
                        if tool_input_display:
                            self._log("   Parameters:")
                            self._log(self._format_json(tool_input_display))
                        else:
                            self._log("   Parameters: {}")
                        self._log("🤖 TOOL RESULT:")
                        self._log(self._format_json(result_dict))
                        self._log("")

                        self._update_cot_step(action_step_num, status="completed", result_summary=self._format_json(result_dict))
                        self._register_tool_success(action_step_num, True)
                        self._record_observation(action_step_num, tool_name, result_dict)

                # Check if done
                if finish_reason == "stop" or not has_tool_calls:
                    # Build a concise summary from assistant content and evidence
                    concise_summary, confidence = self._derive_concise_summary(assistant_text, self.investigation.evidence_trail)

                    self.investigation.root_cause = concise_summary
                    self.investigation.confidence_score = confidence
                    self.investigation.status = "complete"

                    self._print_section("INVESTIGATION COMPLETE")
                    self._log(f"✅ Root Cause Identified (Confidence: {confidence:.2f})")
                    self._log(f"📌 Root Cause Summary: {concise_summary}")
                    self._log("")
                    if self.investigation.evidence_trail:
                        self._print_subsection("EVIDENCE REVIEW")
                        for ev in self.investigation.evidence_trail:
                            tool_label = ev.get("tool", "unknown_tool")
                            self._log(f"• {tool_label}:")
                            self._log(self._format_json(ev.get('result', {})))
                            self._log("")

                    if self._cot_conclusion_step_num is None:
                        parent_for_conclusion = sorted(self._cot_action_steps) if self._cot_action_steps else self._cot_last_added_step
                        self._cot_conclusion_step_num = self._add_cot_step(
                            step_type="conclusion",
                            title="Final Analysis and Conclusion",
                            parent_step=parent_for_conclusion,
                            agent_thoughts=trimmed_text or concise_summary
                        )

                    conclusion_data = {
                        "type": self._slugify(concise_summary),
                        "summary": concise_summary,
                        "confidence": round(confidence, 2),
                        "affected_people": alert.get("metrics", {}).get("team_members_affected", []),
                        "evidence_strength": self._determine_evidence_strength(len(self.investigation.evidence_trail))
                    }

                    self._update_cot_step(
                        self._cot_conclusion_step_num,
                        confidence=round(confidence, 2),
                        confidence_reason="Confidence derived from analysis of gathered evidence.",
                        evidence_gathered=len(self.investigation.evidence_trail),
                        root_cause=conclusion_data
                    )

                    self._finalize_cot_logs(self.investigation, confidence)
                    break

        except Exception as e:
            self._print_subsection("ERROR")
            self._log(f"⚠️  OpenAI API call failed: {e}")
            self._log("   Using mock investigation data for demo...\n")
            mock_investigation = self._create_mock_investigation(investigation_id, alert)
            error_step = self._add_cot_step(
                step_type="reasoning",
                title="OpenAI Failure Handling",
                parent_step=self._cot_last_added_step,
                agent_thoughts=f"OpenAI API call failed: {e}. Using mock investigation data instead."
            )
            self._cot_conclusion_step_num = self._add_cot_step(
                step_type="conclusion",
                title="Fallback Investigation Summary",
                parent_step=error_step,
                agent_thoughts=mock_investigation.root_cause,
                confidence=round(mock_investigation.confidence_score, 2),
                confidence_reason="OpenAI API failure triggered fallback to mock investigation.",
                evidence_gathered=len(mock_investigation.evidence_trail),
                root_cause={
                    "type": self._slugify(mock_investigation.root_cause),
                    "summary": mock_investigation.root_cause,
                    "confidence": round(mock_investigation.confidence_score, 2),
                    "affected_people": alert.get("metrics", {}).get("team_members_affected", []),
                    "evidence_strength": self._determine_evidence_strength(len(mock_investigation.evidence_trail))
                }
            )
            self._finalize_cot_logs(mock_investigation, mock_investigation.confidence_score)
            return mock_investigation

        if not self._cot_payload_saved:
            summary = self.investigation.root_cause or "Investigation incomplete"
            confidence = self.investigation.confidence_score or 0.0
            if self._cot_conclusion_step_num is None:
                parent_for_conclusion = sorted(self._cot_action_steps) if self._cot_action_steps else self._cot_last_added_step
                self._cot_conclusion_step_num = self._add_cot_step(
                    step_type="conclusion",
                    title="Final Analysis and Conclusion",
                    parent_step=parent_for_conclusion,
                    agent_thoughts=summary
                )

            conclusion_data = {
                "type": self._slugify(summary),
                "summary": summary,
                "confidence": round(confidence, 2),
                "affected_people": self._cot_alert_snapshot.get("metrics", {}).get("team_members_affected", []),
                "evidence_strength": self._determine_evidence_strength(len(self.investigation.evidence_trail))
            }

            self._update_cot_step(
                self._cot_conclusion_step_num,
                confidence=round(confidence, 2),
                confidence_reason="Auto-generated after investigation loop end.",
                evidence_gathered=len(self.investigation.evidence_trail),
                root_cause=conclusion_data
            )
            self._finalize_cot_logs(self.investigation, confidence)

        return self.investigation


# For testing
if __name__ == "__main__":
    from anomaly_detector import AnomalyDetector

    # First get an alert
    detector = AnomalyDetector()
    alert = detector.check_for_anomalies()

    if alert:
        # Then investigate
        investigator = RootCauseInvestigator()
        investigation = investigator.investigate(alert)

        print(f"\n📋 Investigation Summary:")
        print(f"   ID: {investigation.investigation_id}")
        print(f"   Status: {investigation.status}")
        print(f"   Confidence: {investigation.confidence_score:.2%}")
        print(f"   Evidence Trail Length: {len(investigation.evidence_trail)}")

