"""
Main Orchestrator - Coordinates all agents in the system
Ties together: Anomaly Detector → Investigator → Planner → Supervisor → Monitor
"""

import json
import time
from datetime import datetime
from anomaly_detector import AnomalyDetector
from root_cause_investigator import RootCauseInvestigator
from intervention_planner import InterventionPlanner
from supervisor_agent import SupervisorAgent
from outcome_monitor import OutcomeMonitor
from typing import Dict


class ProductivityAgentSystem:
    """
    Complete multi-agent system for investigating and resolving productivity issues.

    Flow:
    1. AnomalyDetector: Detects problems
    2. RootCauseInvestigator: Investigates why
    3. InterventionPlanner: Plans solution
    4. SupervisorAgent: Validates and executes
    5. OutcomeMonitor: Tracks results
    """

    def __init__(self):
        self.detector = AnomalyDetector()
        self.investigator = RootCauseInvestigator()
        self.planner = InterventionPlanner()
        self.supervisor = SupervisorAgent()
        self.monitor = OutcomeMonitor()

        self.run_log = []

    def log_event(self, component: str, event: str, details: dict = None):
        """Log system events"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "component": component,
            "event": event,
            "details": details or {}
        }
        self.run_log.append(log_entry)

    def run_full_cycle(self) -> Dict:
        """
        Run complete cycle:
        Anomaly → Investigation → Planning → Supervision → Monitoring
        """
        print("\n" + "=" * 80)
        print("🚀 PRODUCTIVITY AGENT SYSTEM - FULL CYCLE START")
        print("=" * 80)

        start_time = datetime.utcnow()
        cycle_result = {
            "start_time": start_time.isoformat(),
            "phases": {},
            "status": "in_progress"
        }

        # PHASE 1: ANOMALY DETECTION
        print("\n\n" + "🔴" * 40)
        print("PHASE 1: ANOMALY DETECTION")
        print("🔴" * 40)

        alert = self.detector.check_for_anomalies()

        if not alert:
            print("\n✅ No anomalies detected. System healthy.")
            cycle_result["status"] = "no_anomaly"
            return cycle_result

        self.log_event("anomaly_detector", "anomaly_detected", alert)
        cycle_result["phases"]["anomaly_detection"] = {
            "alert": alert,
            "severity": alert.get("severity")
        }

        # PHASE 2: ROOT CAUSE INVESTIGATION
        print("\n\n" + "🟠" * 40)
        print("PHASE 2: ROOT CAUSE INVESTIGATION")
        print("🟠" * 40)

        investigation = self.investigator.investigate(alert)

        self.log_event("investigator", "investigation_complete", {
            "investigation_id": investigation.investigation_id,
            "confidence": investigation.confidence_score
        })
        cycle_result["phases"]["investigation"] = {
            "investigation_id": investigation.investigation_id,
            "root_cause": investigation.root_cause[:200] if investigation.root_cause else None,
            "confidence": investigation.confidence_score,
            "evidence_count": len(investigation.evidence_trail)
        }

        # PHASE 3: INTERVENTION PLANNING
        print("\n\n" + "🟡" * 40)
        print("PHASE 3: INTERVENTION PLANNING")
        print("🟡" * 40)

        plan