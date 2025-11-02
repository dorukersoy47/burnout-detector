"""
Root Cause Investigator Agent - The main agentic component
Autonomously investigates root causes through hypothesis testing.

Agentic Score: 95% - Full autonomous investigation with dynamic tool selection
"""

import json
from openai import OpenAI
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
from config import Config


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
            return self._create_mock_investigation(investigation_id, alert)

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
                if assistant_text.strip():
                    self._print_section(f"STEP {self.step_counter}: AGENT REASONING")
                    self._log("🧠 AGENT THINKS:")
                    self._log(assistant_text.strip())
                    self._log("")

                # Append assistant text to conversation (but do NOT print raw tool dumps)
                assistant_entry = {
                    "role": "assistant",
                    "content": assistant_text
                }

                if getattr(assistant_message, "tool_calls", None):
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

                # If assistant indicated tool calls, process them silently (no big JSON prints)
                if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
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

                # Check if done
                if finish_reason == "stop" or not (hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls):
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
                    break

        except Exception as e:
            self._print_subsection("ERROR")
            self._log(f"⚠️  OpenAI API call failed: {e}")
            self._log("   Using mock investigation data for demo...\n")
            return self._create_mock_investigation(investigation_id, alert)

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

