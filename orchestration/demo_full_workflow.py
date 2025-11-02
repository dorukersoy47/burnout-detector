"""
Demo script - Shows the complete system workflow end-to-end
Runs through: Anomaly → Investigation → Planning → Supervision → Monitoring
"""

import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.anomaly_detector import AnomalyDetector
from agents.root_cause_investigator import RootCauseInvestigator
from agents.intervention_planner import InterventionPlanner
from agents.supervisor_agent import SupervisorAgent
from agents.outcome_monitor import OutcomeMonitor



def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "🔷" * 40)
    print(f"   {title}")
    print("🔷" * 40)


def demo_complete_workflow():
    """Run a complete workflow demonstration"""

    print("\n" + "=" * 80)
    print("🎬 PRODUCTIVITY AGENT SYSTEM - COMPLETE WORKFLOW DEMO")
    print("=" * 80)

    # ==================== PHASE 1: ANOMALY DETECTION ====================
    print_section("PHASE 1: ANOMALY DETECTION")

    detector = AnomalyDetector()
    alert = detector.check_for_anomalies()

    if not alert:
        print("No anomalies detected. Exiting.")
        return

    print(f"\n✅ Alert Generated!")
    print(f"   ID: {alert['timestamp']}")
    print(f"   Severity: {alert['severity']}")
    print(f"   Description: {alert['description']}")

    # ==================== PHASE 2: ROOT CAUSE INVESTIGATION ====================
    print_section("PHASE 2: ROOT CAUSE INVESTIGATION")

    investigator = RootCauseInvestigator()

    try:
        investigation = investigator.investigate(alert)

        print(f"\n✅ Investigation Complete!")
        print(f"   ID: {investigation.investigation_id}")
        print(f"   Confidence: {investigation.confidence_score:.1%}")
        print(f"   Evidence Gathered: {len(investigation.evidence_trail)} data points")
        print(f"   Status: {investigation.status}")

        # Clean up the root cause text - remove technical markers
        root_cause = investigation.root_cause or "Unknown"
        # Remove markdown headers, hypothesis numbers, etc.
        root_cause_lines = [line.strip() for line in root_cause.split('\n') if line.strip() and not line.startswith('#')]
        # Take first meaningful paragraph
        clean_root_cause = ' '.join(root_cause_lines[:3]) if root_cause_lines else root_cause
        if len(clean_root_cause) > 200:
            clean_root_cause = clean_root_cause[:200] + "..."

        print(f"\n📌 Root Cause Found:")
        print(f"   {clean_root_cause}\n")

    except Exception as e:
        print(f"\n⚠️  Investigation skipped due to API issue: {str(e)}")
        print("   Using mock investigation data for demo...\n")

        # Create mock investigation for demo purposes
        investigation = type('Investigation', (), {
            'investigation_id': 'inv_20251101_mock',
            'root_cause': 'New daily 2-hour strategy meetings started, blocking senior engineers (alice, bob) from code reviews',
            'confidence_score': 0.89,
            'evidence_trail': [
                {'tool': 'github_get_pr_metrics', 'finding': 'Review time: 8h → 48h'},
                {'tool': 'calendar_get_events', 'finding': 'New meeting started 2025-10-24'},
                {'tool': 'slack_get_messages', 'finding': 'Team morale: negative'}
            ],
            'status': 'complete'
        })()

    # ==================== PHASE 3: INTERVENTION PLANNING ====================
    print_section("PHASE 3: INTERVENTION PLANNING")

    planner = InterventionPlanner()
    plan = planner.plan(investigation.root_cause, {})

    print(f"\n✅ Intervention Plan Generated!")

    if "options" in plan:
        print(f"\n📋 Options Evaluated: {len(plan['options'])}")
        for i, option in enumerate(plan['options'][:3], 1):
            print(f"\n   Option {i}: {option.get('name', 'Unknown')}")
            print(f"   - Expected Improvement: {option.get('expected_improvement', 0):.0%}")
            print(f"   - Implementation: {option.get('implementation_days', 0)} days")
            print(f"   - Success Rate (historical): {option.get('success_rate_from_history', 0):.0%}")
            print(f"   - Cost: {option.get('cost', 'unknown')}")

    print(f"\n🎯 Recommendation: {plan.get('recommendation', 'Unknown')}")
    print(f"   Confidence: {plan.get('confidence', 0):.0%}")

    # ==================== PHASE 4: SUPERVISOR & APPROVAL ====================
    print_section("PHASE 4: VALIDATION & APPROVAL")

    supervisor = SupervisorAgent()

    # Create a mock investigation dict for supervisor
    inv_dict = {
        "investigation_id": investigation.investigation_id,
        "confidence_score": investigation.confidence_score,
        "root_cause": investigation.root_cause,
        "evidence_trail": investigation.evidence_trail
    }

    orchestration = supervisor.orchestrate(inv_dict, plan)

    validation_result = orchestration.get("validation", {})
    validation_passed = validation_result.get("passed", False)

    print(f"\n✅ Orchestration Complete!")
    print(f"   Validation: {'PASSED' if validation_passed else 'FAILED'}")

    if not validation_passed:
        failure_reason = orchestration.get("reason", "validation_failed")
        print(f"   Orchestration stopped early (reason: {failure_reason})")
    else:
        risk_level = orchestration.get("risk_level", "unknown")
        approval = orchestration.get("approval", {})
        execution = orchestration.get("execution", {})

        print(f"   Risk Level: {risk_level.upper()}")
        print(f"   Approver: {approval.get('approver', 'unknown').upper()}")
        print(f"   Approval Status: {approval.get('status', 'pending').upper()}")
        print(f"   Execution Status: {execution.get('status', 'pending').upper()}")

        if execution.get('actions_taken', 0) > 0:
            print(f"\n📋 Actions Executed: {execution['actions_taken']}")
            for action in execution.get('action_details', [])[:3]:
                print(f"   {action}")

    # ==================== PHASE 5: OUTCOME MONITORING ====================
    print_section("PHASE 5: OUTCOME MONITORING")

    monitor = OutcomeMonitor()

    baseline_metrics = alert.get("metrics", {})
    monitoring_id = monitor.start_monitoring(
        investigation.investigation_id,
        plan,
        baseline_metrics
    )

    print(f"\n✅ Monitoring Started!")
    print(f"   Monitoring ID: {monitoring_id}")

    # Simulate daily checks
    print(f"\n📈 Simulating daily monitoring checks...\n")

    daily_results = []
    for day in range(1, 8, 2):
        daily_check = monitor.monitor_daily()
        daily_results.append(daily_check)

        days = daily_check['days_elapsed']
        productivity = daily_check['latest_metrics']['productivity']
        review_time = daily_check['latest_metrics']['pr_review_time']

        print(f"   Day {days}: Productivity {productivity}/10, Review Time {review_time}h")

    # Generate final report
    print(f"\n📄 Generating final report...\n")
    report = monitor.generate_report()

    print(f"✅ Monitoring Complete!")
    print(f"   Duration: {report['duration_days']} days")
    print(f"   Productivity Improvement: {report['improvements']['productivity_pct']:+.0%}")
    print(f"   Review Time Improvement: {report['improvements']['review_time_pct']:+.0%}")
    print(f"   Overall Success: {'YES ✓' if report['success'] else 'NEEDS ADJUSTMENT'}")

    if report['side_effects']:
        print(f"\n⚠️  Side Effects Detected:")
        for effect in report['side_effects']:
            print(f"   - {effect}")
    else:
        print(f"\n✓ No side effects detected")

    # ==================== FINAL SUMMARY ====================
    print_section("COMPLETE WORKFLOW SUMMARY")

    # Safe summary fields (handle early-stop/validation failure)
    approval_safe = orchestration.get('approval', {}) if validation_passed else {}
    approver_text = (approval_safe.get('approver', 'N/A') or 'N/A').upper()
    risk_text = (orchestration.get('risk_level', 'N/A') if validation_passed else 'N/A').upper()
    execution_status = orchestration.get('execution', {}).get('status', 'skipped' if not validation_passed else 'unknown').upper()
    executed_mark = '✓' if execution_status == 'EXECUTED' or execution_status == 'EXECUTED'.upper() else ('—' if not validation_passed else execution_status)

    print(f"""
    🎯 FULL CYCLE COMPLETED:

    1️⃣  Anomaly Detected: Productivity dropped 30%
    2️⃣  Root Cause Found: {investigation.root_cause[:80]}...
    3️⃣  Solution Planned: {plan.get('recommendation', 'Unknown')}
    4️⃣  Approval Routed: {approver_text} approved
    5️⃣  Intervention Executed: {executed_mark}
    6️⃣  Results Monitored: +{report['improvements']['productivity_pct']:.0%} productivity

    💡 Key Insights:
       - Root cause identified with {investigation.confidence_score:.0%} confidence
       - Solution based on {plan.get('similar_cases', 0)} historical cases
    - Intervention cost: {risk_text} risk
       - Expected ROI: HIGH

    📊 Next Steps:
       - Keep monitoring for next 7 days
       - Celebrate team win! 🎉
       - Update RLHF model with this success
       - Apply learnings to future similar cases
    """)


if __name__ == "__main__":
    demo_complete_workflow()