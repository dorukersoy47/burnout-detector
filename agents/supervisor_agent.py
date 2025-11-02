"""
Supervisor Agent - Validates findings, routes approvals, manages execution
Mostly workflow (20% agentic) but includes governance logic.

Agentic Score: 20% - Rules-based with some reasoning about risk assessment
"""

import json
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ApprovalRequest:
    """Approval request details"""
    investigation_id: str
    root_cause: str
    confidence: float
    risk_level: str
    estimated_cost: str
    approver: str  # who needs to approve
    status: str  # "pending", "approved", "rejected"


class SupervisorAgent:
    """
    Validates investigations, routes for approval, and manages execution.

    Key characteristics:
    - Follows governance rules (deterministic)
    - Validates evidence quality
    - Routes to appropriate approver
    - Manages intervention execution
    """

    def __init__(self):
        self.governance_rules = {
            "high_risk": {"approver": "vp", "min_confidence": 0.90},
            "medium_risk": {"approver": "manager", "min_confidence": 0.80},
            "low_risk": {"approver": "team_lead", "min_confidence": 0.70}
        }
        self.approval_queue = []

    def validate_investigation(self, investigation: Dict) -> Dict:
        """Validate that investigation meets quality standards"""
        print("\n" + "=" * 70)
        print("✅ SUPERVISOR AGENT - Validation")
        print("=" * 70)

        confidence = investigation.get("confidence_score", 0)
        evidence_count = len(investigation.get("evidence_trail", []))

        print(f"\n🔍 Validating Investigation {investigation.get('investigation_id')}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Evidence pieces: {evidence_count}")

        # Validation checks
        checks = {
            "high_confidence": confidence >= 0.75,
            "sufficient_evidence": evidence_count >= 3,
            "root_cause_identified": bool(investigation.get("root_cause")),
            "evidence_trail_complete": evidence_count > 0
        }

        print(f"\n📋 Validation Checklist:")
        for check, passed in checks.items():
            print(f"   {'✓' if passed else '✗'} {check}")

        validation_result = {
            "investigation_id": investigation.get("investigation_id"),
            "passed": all(checks.values()),
            "checks": checks,
            "confidence": confidence,
            "evidence_count": evidence_count
        }

        if all(checks.values()):
            print(f"\n✅ Validation PASSED")
        else:
            print(f"\n❌ Validation FAILED")

        return validation_result

    def assess_risk(self, plan: Dict) -> str:
        """Assess risk level of proposed intervention"""
        print("\n" + "=" * 70)
        print("⚠️ SUPERVISOR AGENT - Risk Assessment")
        print("=" * 70)

        risk_factors = {
            "implementation_cost": plan.get("options", [{}])[0].get("cost", "low"),
            "implementation_days": plan.get("options", [{}])[0].get("implementation_days", 1),
            "risk": plan.get("options", [{}])[0].get("risk", "low")
        }

        print(f"\n🎯 Risk Factors:")
        for factor, value in risk_factors.items():
            print(f"   {factor}: {value}")

        # Determine risk level
        if risk_factors["risk"] == "high" or risk_factors["implementation_days"] > 7:
            risk_level = "high"
        elif risk_factors["risk"] == "medium" or risk_factors["implementation_days"] > 3:
            risk_level = "medium"
        else:
            risk_level = "low"

        print(f"\n🚨 Overall Risk Level: {risk_level.upper()}")

        return risk_level

    def route_for_approval(self, validation: Dict, plan: Dict, risk_level: str) -> ApprovalRequest:
        """Determine who needs to approve and create approval request"""
        print("\n" + "=" * 70)
        print("🔀 SUPERVISOR AGENT - Approval Routing")
        print("=" * 70)

        # Get governance rule
        gov_rule = self.governance_rules.get(f"{risk_level}_risk", {})
        approver = gov_rule.get("approver", "manager")
        min_confidence = gov_rule.get("min_confidence", 0.80)

        confidence = validation.get("confidence", 0)
        passed_validation = validation.get("passed", False)

        print(f"\n📧 Routing Decision:")
        print(f"   Risk Level: {risk_level}")
        print(f"   Approver Required: {approver.upper()}")
        print(f"   Min Confidence Threshold: {min_confidence:.0%}")
        print(f"   Current Confidence: {confidence:.2%}")
        print(f"   Validation Passed: {'Yes' if passed_validation else 'No'}")

        # Create approval request
        approval_request = ApprovalRequest(
            investigation_id=validation.get("investigation_id"),
            root_cause=plan.get("recommendation", "Unknown"),
            confidence=confidence,
            risk_level=risk_level,
            estimated_cost=plan.get("options", [{}])[0].get("cost", "unknown"),
            approver=approver,
            status="pending"
        )

        print(f"\n✉️ Approval Request Created")
        print(f"   To: {approver}")
        print(f"   Status: {approval_request.status}")

        self.approval_queue.append(approval_request)
        return approval_request

    def simulate_approval(self, approval_request: ApprovalRequest) -> bool:
        """Simulate manager/VP approval (in real system, this would be async)"""
        print(f"\n⏳ Awaiting {approval_request.approver.upper()} approval...")
        print(f"   Request sent to: {approval_request.approver}@company.com")

        # Simulate approval
        approved = approval_request.confidence >= 0.75

        if approved:
            print(f"   ✅ APPROVED!")
            approval_request.status = "approved"
        else:
            print(f"   ❌ REJECTED - Low confidence")
            approval_request.status = "rejected"

        return approved

    def execute_intervention(self, plan: Dict, approved: bool) -> Dict:
        """Execute the approved intervention"""
        print("\n" + "=" * 70)
        print("⚙️ SUPERVISOR AGENT - Execution")
        print("=" * 70)

        if not approved:
            print("\n❌ Execution SKIPPED - Not approved")
            return {"status": "skipped", "reason": "not_approved"}

        print("\n▶️ Executing Approved Intervention")

        # Get recommended intervention
        recommendation = plan.get("recommendation", "")

        print(f"\n🔧 Executing: {recommendation}")

        # Simulate execution actions
        actions = []

        if "async" in recommendation.lower():
            actions = [
                "✓ Created Slack channel: #pr-review-updates",
                "✓ Scheduled async standup: 9am daily async thread",
                "✓ Notified alice and bob about new process",
                "✓ Updated team wiki with new guidelines",
                "✓ Archived old daily meeting"
            ]
        elif "reduce" in recommendation.lower():
            actions = [
                "✓ Rescheduled Wednesday meeting to async",
                "✓ Sent calendar invite updates",
                "✓ Notified team of change",
                "✓ Set up async alternative"
            ]
        else:
            actions = [
                "✓ Recorded intervention action",
                "✓ Notified stakeholders",
                "✓ Set up monitoring"
            ]

        for action in actions:
            print(f"   {action}")

        execution_result = {
            "status": "executed",
            "recommendation": recommendation,
            "actions_taken": len(actions),
            "execution_timestamp": datetime.utcnow().isoformat(),
            "action_details": actions
        }

        print(f"\n✅ Execution Complete")

        return execution_result

    def orchestrate(self, investigation: Dict, plan: Dict) -> Dict:
        """Main orchestration flow"""
        print("\n" + "=" * 70)
        print("🎯 SUPERVISOR AGENT - Full Orchestration")
        print("=" * 70)

        # Step 1: Validate
        validation = self.validate_investigation(investigation)

        if not validation["passed"]:
            return {
                "status": "rejected",
                "reason": "validation_failed",
                "validation": validation
            }

        # Step 2: Assess risk
        risk_level = self.assess_risk(plan)

        # Step 3: Route for approval
        approval_request = self.route_for_approval(validation, plan, risk_level)

        # Step 4: Simulate approval
        approved = self.simulate_approval(approval_request)

        # Step 5: Execute
        execution = self.execute_intervention(plan, approved)

        # Return final orchestration result
        return {
            "status": "complete",
            "validation": validation,
            "risk_level": risk_level,
            "approval": {
                "approver": approval_request.approver,
                "status": approval_request.status
            },
            "execution": execution
        }


# For testing
if __name__ == "__main__":
    supervisor = SupervisorAgent()

    # Mock investigation
    investigation = {
        "investigation_id": "inv_20251101_195849",
        "confidence_score": 0.89,
        "root_cause": "New daily meetings blocking code reviews",
        "evidence_trail": [{}, {}, {}, {}]
    }

    # Mock plan
    plan = {
        "options": [{
            "name": "Async Updates",
            "cost": "low",
            "implementation_days": 1,
            "risk": "low"
        }],
        "recommendation": "Convert to async updates"
    }

    result = supervisor.orchestrate(investigation, plan)

    print(f"\n📋 Orchestration Result:")
    print(json.dumps(result, indent=2))