"""
Intervention Planner Agent - Recommends solutions based on historical patterns
Uses RLHF-style learning from past outcomes.

Agentic Score: 60% - Learns from patterns, contextualizes recommendations
"""

import json
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from config import Config


@dataclass
class HistoricalCase:
    """Represents a past intervention case"""
    case_id: str
    root_cause: str
    intervention: str
    outcome: str  # "success", "partial", "failed"
    success_metric_improvement: float
    team_size: int
    days_to_effect: int
    notes: str


class InterventionPlanner:
    """
    Plans interventions based on historical patterns.

    Key agentic behaviors:
    - Searches historical cases
    - Learns success rates from past
    - Contextualizes recommendations
    - Reasons about tradeoffs
    - Predicts outcomes
    """

    # NOTE: For the current MVP some supporting tools return hardcoded mock data
    #       (see entries under tools/) instead of making live API calls. This is
    #       acceptable for the hackathon demo, but expanding beyond the MVP will
    #       require fleshing out those integrations (Slack, calendar, etc.) with
    #       real query() methods and API-backed responses.

    def __init__(self):
        # Get API key from config
        if Config.is_openai_configured():
            try:
                self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
                print("✅ OpenAI client initialized (API key from config)")
            except Exception as e:
                print(f"⚠️ OpenAI client init failed: {e}")
                self.client = None
        else:
            print("⚠️ No OPENAI_API_KEY set in config.py — OpenAI client disabled, using local fallback")
            self.client = None

        self.model = Config.OPENAI_MODEL
        self.historical_cases = self._load_historical_cases()

    def _load_historical_cases(self) -> List[HistoricalCase]:
        """Load dummy historical data"""
        return [
            HistoricalCase(
                case_id="hist_001",
                root_cause="Meeting overload blocking code review",
                intervention="Convert daily meeting to async updates",
                outcome="success",
                success_metric_improvement=0.37,
                team_size=8,
                days_to_effect=3,
                notes="Senior engineers freed up 2h/day"
            ),
            HistoricalCase(
                case_id="hist_002",
                root_cause="Meeting overload blocking code review",
                intervention="Reduce meeting frequency (2x/week instead of daily)",
                outcome="success",
                success_metric_improvement=0.25,
                team_size=6,
                days_to_effect=2,
                notes="Quick win, good team buy-in"
            ),
            HistoricalCase(
                case_id="hist_003",
                root_cause="Meeting overload blocking code review",
                intervention="Hire additional reviewer",
                outcome="partial",
                success_metric_improvement=0.15,
                team_size=12,
                days_to_effect=14,
                notes="Took time to ramp up"
            ),
            HistoricalCase(
                case_id="hist_004",
                root_cause="Burnout causing slow reviews",
                intervention="Enforce no-meeting Fridays",
                outcome="success",
                success_metric_improvement=0.22,
                team_size=9,
                days_to_effect=7,
                notes="Gradual improvement, morale boost"
            ),
            HistoricalCase(
                case_id="hist_005",
                root_cause="Technical debt blocking velocity",
                intervention="Sprint dedicated to tech debt cleanup",
                outcome="success",
                success_metric_improvement=0.40,
                team_size=5,
                days_to_effect=10,
                notes="Significant long-term benefit"
            ),
            HistoricalCase(
                case_id="hist_006",
                root_cause="Meeting overload blocking code review",
                intervention="Hire additional reviewer",
                outcome="success",
                success_metric_improvement=0.32,
                team_size=8,
                days_to_effect=7,
                notes="Quick ramp, strong performer"
            ),
            HistoricalCase(
                case_id="hist_007",
                root_cause="Meeting overload blocking code review",
                intervention="Shift to async-first culture",
                outcome="success",
                success_metric_improvement=0.29,
                team_size=10,
                days_to_effect=5,
                notes="Requires cultural change"
            ),
        ]

    def find_similar_cases(self, root_cause: str) -> List[Dict]:
        """Find historical cases with similar root causes"""
        similar = []

        for case in self.historical_cases:
            # Simple string matching - in production, use semantic search
            if root_cause.lower() in case.root_cause.lower() or \
                    case.root_cause.lower() in root_cause.lower():
                similar.append({
                    "case_id": case.case_id,
                    "intervention": case.intervention,
                    "outcome": case.outcome,
                    "improvement": case.success_metric_improvement,
                    "days_to_effect": case.days_to_effect
                })

        return similar

    def calculate_success_rates(self, cases: List[Dict]) -> Dict[str, float]:
        """Calculate success rate for each intervention type"""
        interventions = {}

        for case in cases:
            intervention = case["intervention"]
            outcome = case["outcome"]

            if intervention not in interventions:
                interventions[intervention] = {"total": 0, "success": 0}

            interventions[intervention]["total"] += 1
            if outcome == "success":
                interventions[intervention]["success"] += 1

        # Calculate rates
        rates = {}
        for intervention, data in interventions.items():
            rates[intervention] = {
                "success_rate": data["success"] / data["total"],
                "success_count": data["success"],
                "total_count": data["total"]
            }

        return rates

    def plan(self, root_cause: str, investigation_details: Dict) -> Dict:
        """
        Plan intervention using historical patterns and reasoning.
        """
        print("\n" + "=" * 70)
        print("📋 INTERVENTION PLANNER AGENT")
        print("=" * 70)
        print(f"\n🔍 Root Cause: {root_cause}\n")

        # Find similar historical cases
        similar_cases = self.find_similar_cases(root_cause)
        print(f"📚 Found {len(similar_cases)} similar historical cases")

        # Calculate success rates
        success_rates = self.calculate_success_rates(similar_cases)

        print("\n📊 Historical Success Rates:")
        for intervention, rates in success_rates.items():
            print(f"   {intervention}: {rates['success_rate']:.0%} "
                  f"({rates['success_count']}/{rates['total_count']})")

        # Use ChatGPT for reasoning about best approach
        system_prompt = """You are an Intervention Planner Agent. Your job is to:

1. Receive a root cause and historical case data
2. Analyze what worked in the past (and what didn't)
3. Generate 3-4 intervention options with pros/cons
4. Consider tradeoffs: success rate vs. implementation difficulty vs. cost
5. Rank by expected ROI
6. Recommend the best option with reasoning

OUTPUT FORMAT:
Generate a JSON response with:
{
  "options": [
    {
      "name": "Option name",
      "description": "What we would do",
      "expected_improvement": 0.XX (0-1 scale),
      "implementation_days": X,
      "cost": "low/medium/high",
      "risk": "low/medium/high",
      "success_rate_from_history": 0.XX,
      "pros": ["pro1", "pro2"],
      "cons": ["con1", "con2"],
      "roi_score": 0.XX (higher is better)
    }
  ],
  "recommendation": "Option name - because...",
  "confidence": 0.XX
}"""

        user_message = f"""Root Cause: {root_cause}

Historical patterns (similar cases):
{json.dumps(similar_cases, indent=2)}

Success rates by intervention type:
{json.dumps(success_rates, indent=2)}

Generate 3-4 intervention options ranked by ROI."""

        plan_json = None

        # If OpenAI client isn't configured, skip remote call and use fallback
        if not self.client:
            print("\n⚠️ OpenAI client unavailable — skipping ChatGPT call and using local default plan.")
            plan_json = self._generate_default_plan(success_rates)
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=Config.INTERVENTION_PLANNING_TEMPERATURE,
                    max_tokens=2000
                )

                response_text = response.choices[0].message.content

                # Parse response (with fallback) - but don't print the raw JSON
                try:
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        plan_json = json.loads(response_text[json_start:json_end])
                    else:
                        plan_json = self._generate_default_plan(success_rates)
                except json.JSONDecodeError:
                    plan_json = self._generate_default_plan(success_rates)

            except Exception as e:
                # Handle missing API key / network / OpenAI client errors gracefully
                print(f"\n⚠️  ChatGPT call failed: {e}")
                print("   Falling back to local default plan generator.")
                plan_json = self._generate_default_plan(success_rates)

        # Ensure plan_json is always a dict
        if plan_json is None:
            plan_json = self._generate_default_plan(success_rates)

        plan_json["similar_cases"] = len(similar_cases)
        plan_json["generated_at"] = datetime.utcnow().isoformat()

        # Print clean, formatted output instead of raw JSON
        print("\n🎯 Analysis Complete!\n")

        return plan_json

    def _generate_default_plan(self, success_rates: Dict) -> Dict:
        """Fallback plan if ChatGPT doesn't return valid JSON"""
        # Handle empty success_rates to avoid ValueError
        if not success_rates:
            return {
                "options": [
                    {
                        "name": "Conservative: Monitor & Low-risk changes",
                        "description": "No historical matches. Recommend monitoring and low-risk process changes.",
                        "expected_improvement": 0.15,
                        "implementation_days": 2,
                        "cost": "low",
                        "risk": "low",
                        "success_rate_from_history": 0.25,
                        "pros": ["Low disruption", "Quick to implement"],
                        "cons": ["May be less effective"],
                        "roi_score": 0.5
                    }
                ],
                "recommendation": "Conservative: Monitor & Low-risk changes",
                "confidence": 0.40
            }

        best_intervention = max(
            success_rates.items(),
            key=lambda x: x[1]["success_rate"]
        )

        return {
            "options": [
                {
                    "name": best_intervention[0],
                    "description": f"Based on {best_intervention[1]['total_count']} historical cases",
                    "expected_improvement": 0.30,
                    "implementation_days": 3,
                    "cost": "low",
                    "risk": "low",
                    "success_rate_from_history": best_intervention[1]["success_rate"],
                    "pros": ["Proven to work", "Quick implementation"],
                    "cons": ["Requires team buy-in"],
                    "roi_score": 0.85
                }
            ],
            "recommendation": best_intervention[0],
            "confidence": 0.80
        }


# For testing
if __name__ == "__main__":
    planner = InterventionPlanner()

    root_cause = "New daily 2-hour strategy meetings started, blocking senior engineers (alice, bob) from code reviews"

    plan = planner.plan(root_cause, {})

    print(f"\n✅ Plan Generated:")
    print(json.dumps(plan, indent=2))