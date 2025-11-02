# Lighthouse.ai — *“AI agents that investigate why your team's productivity drops—not just that it did.”*  
![Lighthouse.ai Logo Placeholder](docs/assets/logo-placeholder.png)

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license) [![AgentVerse Hackathon](https://img.shields.io/badge/AgentVerse-2025-orange.svg)](#hackathon-context)

---

## 🚨 Problem

Engineering leads lose weeks chasing the root cause of productivity dips. Dashboards show *what* changed—“PR review time up 442%”—but they never answer *why*. Meanwhile, teams burn out, delivery commitments slip, and institutional knowledge fades.

> **Typical spiral:** manual log-dives → meetings with every stakeholder → trial-and-error fixes → repeat next quarter. Lighthouse.ai stops that spiral.

---

## 🛠️ Solution Overview

Lighthouse.ai is an autonomous productivity intelligence system. It detects anomalies, runs a ReAct-style investigation to prove causation, plans targeted interventions, routes approvals, executes changes, and learns from real outcomes.

```
[ Anomaly Detector ] → [ Root Cause Investigator 🤖 ] → [ Intervention Planner ] 
              ↓                                   ↓
    Evidence sources (GitHub, Slack, Calendar, Jira via tools/)
              ↓                                   ↓
[ Supervisor & Approvals ] → [ Outcome Monitor ] → [ RLHF Learning Loop ]
```

---

## ✨ Key Features

- 🧠 **Autonomous Root-Cause Reasoning** – ReAct loop with chain-of-thought transparency  
- 🔍 **Causal Attribution** – Proves *why* metrics changed, not just that they did  
- 📚 **RLHF Memory** – Learns from every intervention to boost success rates  
- 🤝 **Human-in-the-Loop Safety** – Approval routing & risk guardrails  
- 📈 **Outcome Monitoring** – Closes the loop with measurable impact  
- ⚙️ **API-Ready Architecture** – Mock tools today, ready for live integrations tomorrow

---

## 🔄 How It Works

1. **Daily Anomaly Sweep** – `AnomalyDetector` checks productivity stats (Z-scores, baselines).  
2. **Root Cause Investigation** – LLM-powered agent plans hypotheses, calls tools dynamically, and narrates every thought.  
3. **Intervention Planning** – Ranks options using historical cases (RLHF DB).  
4. **Supervisor Orchestration** – Validates evidence, routes approvals, executes recommended changes.  
5. **Outcome Monitoring** – Tracks improvements, detects side effects, and updates success rates.

---

## 🤖 What Makes It Agentic

- Chooses its own investigation path; no static playbook  
- Uses tool outputs to revise hypotheses mid-flight  
- Explains each decision in a readable chain-of-thought log  
- Learns from feedback via the RLHF success database  
- Balances autonomy with human approvals based on risk

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/your-org/lighthouse-ai.git
cd lighthouse-ai

# (Optional) create env
python3 -m venv .venv && source .venv/bin/activate

# Install deps (mock integrations today)
pip install -r requirements.txt
```

If you have an OpenAI API key, add it to `config.py`; otherwise the MVP uses mocked tool responses.

---

## 🚀 Usage Examples

### Run the end-to-end demo
```bash
python orchestration/demo_full_workflow.py
```

### Use individual agents
```python
from agents.anomaly_detector import AnomalyDetector
alert = AnomalyDetector().check_for_anomalies()

from agents.root_cause_investigator import RootCauseInvestigator
investigation = RootCauseInvestigator().investigate(alert)

from agents.intervention_planner import InterventionPlanner
plan = InterventionPlanner().plan(investigation.root_cause, {})
```

---

## 🗂️ Project Structure

```
burnout-detector/
├─ README.md
├─ requirements.txt
├─ config.py                    # Global settings & API keys (mock-friendly)
├─ agents/                      # Autonomous components
│  ├─ anomaly_detector.py       # Daily metric anomaly detection
│  ├─ root_cause_investigator.py# ReAct investigation with verbose reasoning trace
│  ├─ intervention_planner.py   # RLHF-powered recommendation engine
│  ├─ supervisor_agent.py       # Validation, approvals, execution
│  └─ outcome_monitor.py        # Post-intervention tracking & RLHF updates
├─ orchestration/
│  └─ demo_full_workflow.py     # Orchestrated end-to-end storyline for the MVP
├─ tools/
│  └─ GitHub-Tracker/           # Mock data services (ready to swap with live APIs)
├─ motion/
│  └─ eye.py                    # Example of sensor-style data source
├─ demo_data/
│  └─ GitHub/                   # CSVs feeding mock providers
└─ tests/
   └─ test_systems.py           # System-level tests & guardrails
```

**Interaction Map**

- `orchestration/demo_full_workflow.py` stitches the agents together.  
- Each agent pulls data via mock tool adapters under `tools/`.  
- `OutcomeMonitor` writes learnings back to the RLHF store (in-memory today).  
- Tests ensure regressions are caught when swapping mocks for real APIs.

---

## ▶️ Demo Instructions

1. Ensure dependencies are installed (see Quick Start).  
2. (Optional) Configure OpenAI credentials in `config.py`.  
3. Run `python orchestration/demo_full_workflow.py`.  
4. Observe console output: anomaly detection, full investigation trace, planning, supervisor decisions, and monitoring summary.

---

## 🗃️ RLHF Database Schema (MVP)

```text
historical_cases
├─ case_id (str)
├─ root_cause (str)
├─ intervention (str)
├─ outcome ("success" | "partial" | "failed")
├─ success_metric_improvement (float)
├─ team_size (int)
├─ days_to_effect (int)
└─ notes (str)

rlhf_feedback
├─ intervention (str)
├─ outcome (str)
├─ success_rate (float)
└─ updated_at (datetime)
```

Currently stored in-memory within `intervention_planner.py`; ready to migrate to Postgres/Vector DB.

---

## 🧠 Example Investigation Output

```
══════════════════════════════════════════════════════════════
STEP 1: AGENT REASONING
🧠 AGENT THINKS:
"PR review time jumped 5× and 12 PRs blocked. Hypothesis: senior reviewers stuck in meetings."
...
ACTION & EVIDENCE
🛠️  AGENT DECISION 1: calling `calendar_get_events`
   Parameters:
   { "person": "alice", "days": 14 }
🤖 TOOL RESULT:
{ "meeting_load_hours": 28, "note": "New 2-hour strategy meeting began Oct 24" }

🛠️  AGENT DECISION 2: calling `github_get_pr_metrics`
...
✅ Root Cause Identified (Confidence: 0.92)
📌 Root Cause Summary: New daily strategy meeting consuming senior reviewers’ time.
```

---

## 🏆 Why This Wins

- **First-mover causal reasoning** for developer productivity  
- **Transparent agent**—judges see every hypothesis, not just the answer  
- **RLHF loop** compounding value with every success or failure  
- **Built for enterprises**: approvals, audit trail, side-effect monitoring  
- **Pluggable architecture**—swap mocks for real APIs when integrating with customers

---

## 🏁 Hackathon Context

- **Event:** AgentVerse Hackathon 2025  
- **Track:** Reimagine the Workplace  
- **Focus:** Autonomous agents boosting team productivity & wellbeing

---

## 🛣️ Roadmap

- 🔌 Production-grade GitHub, Slack, Jira, and Calendar connectors  
- 🧠 Organization-specific RLHF tuning  
- 🔮 Predictive anomaly alerts (before metrics crater)  
- 📊 Web dashboard + Slack concierge bot  
- 🌐 Multi-team rollups and cross-team insights

---

## 🤝 Contributing

1. Fork the repo and create a feature branch.  
2. Follow the existing module layout (`agents/`, `orchestration/`, `tools/`).  
3. Add/Update tests in `tests/test_systems.py`.  
4. Submit a PR with a helpful summary and screenshots/logs of your agent trace.

We’re especially looking for contributors who can:  
- Implement real connectors in `tools/` (`slack_client.py`, `calendar_client.py`, etc.).  
- Expand RLHF persistence beyond in-memory structures.  
- Create a web dashboard or Slack interface.

---

## 👥 Team & Contact

- *Team Lighthouse* — AgentVerse Hackathon finalists  
- 📧 Contact: team@lighthouse.ai (placeholder)  
- 🌐 Website: https://lighthouse.ai (placeholder)  
- LinkedIn, Twitter, and demo video placeholders to be added

---

## 📸 Screenshots & Demo Video (placeholders)

- ![Investigation Trace Screenshot](docs/assets/screenshots/investigation.png)  
- [🎥 Demo Video](https://youtu.be/placeholder)

> *Replace with real media before launch.*

---

## 📜 License

This project is licensed under the MIT License – see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- AgentVerse Hackathon organizers and mentors  
- Open-source contributors who inspired our ReAct implementation  
- Mock data sets courtesy of the team’s internal playbooks

---

> **Ready for takeoff:** Lighthouse.ai is already illuminating the dark corners of productivity loss. Let’s bring causality-driven insight to every engineering org.

# burnout-detector
