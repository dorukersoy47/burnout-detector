"""
Comprehensive test suite for the Productivity Agent System
Tests all components: Detector, Investigator, Planner, Supervisor, Monitor
"""

import sys
import json
import unittest
from io import StringIO
from datetime import datetime, timedelta

# Import all components
from anomaly_detector import AnomalyDetector, ProductivityMetrics
from root_cause_investigator import RootCauseInvestigator, Investigation, ToolExecutor
from intervention_planner import InterventionPlanner, HistoricalCase
from supervisor_agent import SupervisorAgent, ApprovalRequest
from outcome_monitor import OutcomeMonitor, MonitoringDataPoint


class TestAnomalyDetector(unittest.TestCase):
    """Test the Anomaly Detector component"""

    def setUp(self):
        self.detector = AnomalyDetector(threshold_z_score=1.5)

    def test_detector_initialization(self):
        """Test that detector initializes with correct parameters"""
        self.assertEqual(self.detector.threshold_z_score, 1.5)
        self.assertEqual(self.detector.baseline_window_days, 30)
        self.assertIsNotNone(self.detector.metrics_history)

    def test_get_dummy_metrics(self):
        """Test dummy metrics generation"""
        metrics = self.detector.get_dummy_metrics(days_ago=0)

        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics.productivity_score, float)
        self.assertGreaterEqual(metrics.productivity_score, 0)
        self.assertLessEqual(metrics.productivity_score, 10)
        self.assertIsInstance(metrics.pr_review_time_hours, float)
        self.assertGreater(metrics.pr_review_time_hours, 0)

    def test_baseline_loading(self):
        """Test baseline metrics loading"""
        baseline = self.detector.load_baseline()

        self.assertIn("productivity_mean", baseline)
        self.assertIn("productivity_stdev", baseline)
        self.assertIn("review_time_mean", baseline)
        self.assertIn("review_time_stdev", baseline)

        # Baseline should have reasonable values
        self.assertGreater(baseline["productivity_mean"], 0)
        self.assertGreaterEqual(baseline["productivity_stdev"], 0)

    def test_z_score_calculation(self):
        """Test z-score calculation"""
        z_score = self.detector.calculate_z_score(
            value=10,
            mean=5,
            stdev=2
        )

        # Z-score of 2.5 for value 10, mean 5, stdev 2
        self.assertAlmostEqual(z_score, 2.5, places=1)

    def test_z_score_with_zero_stdev(self):
        """Test z-score with zero standard deviation"""
        z_score = self.detector.calculate_z_score(
            value=10,
            mean=5,
            stdev=0
        )

        self.assertEqual(z_score, 0)

    def test_anomaly_detection_trigger(self):
        """Test that anomaly is detected when threshold is exceeded"""
        alert = self.detector.check_for_anomalies()

        # Should detect anomaly in dummy data (which shows anomaly)
        self.assertIsNotNone(alert)
        self.assertIn("timestamp", alert)
        self.assertIn("severity", alert)
        self.assertIn("metrics", alert)
        self.assertIn("z_scores", alert)
        self.assertGreater(len(alert.get("affected_areas", [])), 0)

    def test_alert_message_generation(self):
        """Test alert message generation"""
        metrics = self.detector.get_dummy_metrics(days_ago=0)
        baseline = self.detector.load_baseline()

        message = self.detector._generate_alert_message(metrics, baseline)

        self.assertIsInstance(message, str)
        self.assertGreater(len(message), 0)
        self.assertIn("%", message)  # Should contain percentage changes

    def test_affected_areas_identification(self):
        """Test identification of affected areas"""
        metrics = self.detector.get_dummy_metrics(days_ago=0)
        areas = self.detector._identify_affected_areas(metrics)

        self.assertIsInstance(areas, list)
        # Anomaly data should have some affected areas
        self.assertGreater(len(areas), 0)


class TestToolExecutor(unittest.TestCase):
    """Test the Tool Executor (used by Investigator)"""

    def setUp(self):
        self.executor = ToolExecutor()

    def test_tool_executor_initialization(self):
        """Test tool executor initializes correctly"""
        self.assertIsNotNone(self.executor.call_log)
        self.assertEqual(len(self.executor.call_log), 0)

    def test_github_pr_metrics_tool(self):
        """Test GitHub PR metrics tool"""
        result_str = self.executor.execute_tool(
            "github_get_pr_metrics",
            {"repo": "org/repo", "days": 7}
        )

        result = json.loads(result_str)

        self.assertIn("repo", result)
        self.assertIn("avg_review_time_hours", result)
        self.assertIn("top_reviewers", result)
        self.assertIsInstance(result["top_reviewers"], list)

    def test_calendar_events_tool(self):
        """Test calendar events tool"""
        result_str = self.executor.execute_tool(
            "calendar_get_events",
            {"person": "alice", "days": 7}
        )

        result = json.loads(result_str)

        self.assertIn("person", result)
        self.assertIn("events", result)
        self.assertIn("meeting_load_hours", result)
        self.assertIsInstance(result["events"], list)

    def test_slack_messages_tool(self):
        """Test Slack messages tool"""
        result_str = self.executor.execute_tool(
            "slack_get_messages",
            {"channel": "general", "query": "burnout", "days": 7}
        )

        result = json.loads(result_str)

        self.assertIn("channel", result)
        self.assertIn("messages_found", result)
        self.assertIn("messages", result)
        self.assertIn("sentiment", result)

    def test_github_issues_tool(self):
        """Test GitHub issues tool"""
        result_str = self.executor.execute_tool(
            "github_get_issues",
            {"repo": "org/repo"}
        )

        result = json.loads(result_str)

        self.assertIn("repo", result)
        self.assertIn("open_issues", result)
        self.assertIn("critical_issues", result)

    def test_tool_call_logging(self):
        """Test that tool calls are logged"""
        self.executor.execute_tool(
            "github_get_pr_metrics",
            {"repo": "org/repo", "days": 7}
        )

        self.assertEqual(len(self.executor.call_log), 1)
        log_entry = self.executor.call_log[0]

        self.assertIn("tool", log_entry)
        self.assertIn("input", log_entry)
        self.assertIn("timestamp", log_entry)
        self.assertEqual(log_entry["tool"], "github_get_pr_metrics")


class TestRootCauseInvestigator(unittest.TestCase):
    """Test the Root Cause Investigator component"""

    def setUp(self):
        self.investigator = RootCauseInvestigator()

    def test_investigator_initialization(self):
        """Test investigator initializes correctly"""
        self.assertIsNotNone(self.investigator.client)
        self.assertIsNotNone(self.investigator.tool_executor)
        self.assertIsNotNone(self.investigator.tools)
        self.assertGreater(len(self.investigator.tools), 0)

    def test_investigator_tools_schema(self):
        """Test that all tools have proper schema"""
        for tool in self.investigator.tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("input_schema", tool)

            schema = tool["input_schema"]
            self.assertIn("type", schema)
            self.assertIn("properties", schema)
            self.assertEqual(schema["type"], "object")

    def test_investigation_creation(self):
        """Test investigation object creation"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "high",
            "metrics": {},
            "description": "Test alert"
        }

        # We'll test without calling Claude to avoid API costs
        inv = Investigation(
            investigation_id="test_inv_001",
            alert=alert,
            hypotheses=[],
            root_cause=None,
            confidence_score=0.0,
            evidence_trail=[],
            status="in_progress"
        )

        self.assertEqual(inv.investigation_id, "test_inv_001")
        self.assertEqual(inv.status, "in_progress")
        self.assertEqual(len(inv.hypotheses), 0)
        self.assertEqual(len(inv.evidence_trail), 0)


class TestInterventionPlanner(unittest.TestCase):
    """Test the Intervention Planner component"""

    def setUp(self):
        self.planner = InterventionPlanner()

    def test_planner_initialization(self):
        """Test planner initializes correctly"""
        self.assertIsNotNone(self.planner.historical_cases)
        self.assertGreater(len(self.planner.historical_cases), 0)

    def test_historical_cases_loading(self):
        """Test historical cases are loaded correctly"""
        cases = self.planner.historical_cases

        for case in cases:
            self.assertIsInstance(case, HistoricalCase)
            self.assertIn("meeting", case.root_cause.lower() or
                          "burnout" in case.root_cause.lower() or
                          "debt" in case.root_cause.lower())

    def test_find_similar_cases(self):
        """Test finding similar historical cases"""
        root_cause = "Meeting overload blocking code review"
        similar = self.planner.find_similar_cases(root_cause)

        self.assertIsInstance(similar, list)
        self.assertGreater(len(similar), 0)

        for case in similar:
            self.assertIn("case_id", case)
            self.assertIn("intervention", case)
            self.assertIn("outcome", case)

    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        cases = [
            {
                "intervention": "Async Updates",
                "outcome": "success"
            },
            {
                "intervention": "Async Updates",
                "outcome": "success"
            },
            {
                "intervention": "Async Updates",
                "outcome": "failed"
            }
        ]

        rates = self.planner.calculate_success_rates(cases)

        self.assertIn("Async Updates", rates)
        self.assertAlmostEqual(rates["Async Updates"]["success_rate"], 2 / 3, places=2)
        self.assertEqual(rates["Async Updates"]["success_count"], 2)
        self.assertEqual(rates["Async Updates"]["total_count"], 3)

    def test_default_plan_generation(self):
        """Test default plan generation (fallback)"""
        success_rates = {
            "Option A": {"success_rate": 0.75, "total_count": 4},
            "Option B": {"success_rate": 0.50, "total_count": 4}
        }

        plan = self.planner._generate_default_plan(success_rates)

        self.assertIn("options", plan)
        self.assertIn("recommendation", plan)
        self.assertIn("confidence", plan)
        self.assertGreater(len(plan["options"]), 0)


class TestSupervisorAgent(unittest.TestCase):
    """Test the Supervisor Agent component"""

    def setUp(self):
        self.supervisor = SupervisorAgent()

    def test_supervisor_initialization(self):
        """Test supervisor initializes correctly"""
        self.assertIsNotNone(self.supervisor.governance_rules)
        self.assertIsNotNone(self.supervisor.approval_queue)
        self.assertEqual(len(self.supervisor.approval_queue), 0)

    def test_governance_rules(self):
        """Test governance rules are set up"""
        rules = self.supervisor.governance_rules

        self.assertIn("high_risk", rules)
        self.assertIn("medium_risk", rules)
        self.assertIn("low_risk", rules)

        for risk_level, rule in rules.items():
            self.assertIn("approver", rule)
            self.assertIn("min_confidence", rule)

    def test_investigation_validation_pass(self):
        """Test validation passes for good investigation"""
        investigation = {
            "investigation_id": "inv_001",
            "confidence_score": 0.89,
            "root_cause": "Meeting overload",
            "evidence_trail": [{}, {}, {}, {}]
        }

        validation = self.supervisor.validate_investigation(investigation)

        self.assertIn("passed", validation)
        self.assertIn("checks", validation)
        self.assertTrue(validation["passed"])

    def test_investigation_validation_fail(self):
        """Test validation fails for weak investigation"""
        investigation = {
            "investigation_id": "inv_002",
            "confidence_score": 0.45,  # Too low
            "root_cause": None,
            "evidence_trail": []  # No evidence
        }

        validation = self.supervisor.validate_investigation(investigation)

        self.assertFalse(validation["passed"])

    def test_risk_assessment_low(self):
        """Test risk assessment for low-risk intervention"""
        plan = {
            "options": [{
                "cost": "low",
                "implementation_days": 1,
                "risk": "low"
            }]
        }

        risk = self.supervisor.assess_risk(plan)

        self.assertEqual(risk, "low")

    def test_risk_assessment_high(self):
        """Test risk assessment for high-risk intervention"""
        plan = {
            "options": [{
                "cost": "high",
                "implementation_days": 14,
                "risk": "high"
            }]
        }

        risk = self.supervisor.assess_risk(plan)

        self.assertEqual(risk, "high")

    def test_approval_routing(self):
        """Test approval routing"""
        validation = {
            "investigation_id": "inv_001",
            "passed": True,
            "confidence": 0.89
        }

        plan = {
            "options": [{
                "cost": "low",
                "implementation_days": 1,
                "risk": "low"
            }]
        }

        approval = self.supervisor.route_for_approval(
            validation,
            plan,
            "low"
        )

        self.assertIsInstance(approval, ApprovalRequest)
        self.assertEqual(approval.status, "pending")
        self.assertIsNotNone(approval.approver)

    def test_approval_simulation(self):
        """Test approval simulation"""
        approval = ApprovalRequest(
            investigation_id="inv_001",
            root_cause="Test",
            confidence=0.89,
            risk_level="low",
            estimated_cost="low",
            approver="manager",
            status="pending"
        )

        approved = self.supervisor.simulate_approval(approval)

        self.assertTrue(approved)
        self.assertEqual(approval.status, "approved")

    def test_execution(self):
        """Test intervention execution"""
        plan = {
            "recommendation": "Convert to async updates",
            "options": [{
                "cost": "low",
                "implementation_days": 1,
                "risk": "low"
            }]
        }

        execution = self.supervisor.execute_intervention(plan, approved=True)

        self.assertEqual(execution["status"], "executed")
        self.assertIn("actions_taken", execution)
        self.assertGreater(execution["actions_taken"], 0)


class TestOutcomeMonitor(unittest.TestCase):
    """Test the Outcome Monitor component"""

    def setUp(self):
        self.monitor = OutcomeMonitor()

    def test_monitor_initialization(self):
        """Test monitor initializes correctly"""
        self.assertIsNone(self.monitor.monitoring_id)
        self.assertIsNone(self.monitor.intervention_details)
        self.assertEqual(len(self.monitor.monitoring_data), 0)

    def test_start_monitoring(self):
        """Test starting monitoring"""
        intervention = {
            "recommendation": "Async updates"
        }

        baseline = {
            "productivity_score": 6.0,
            "pr_review_time_hours": 48,
            "pr_merge_time_hours": 72,
            "blocked_prs": 12
        }

        monitoring_id = self.monitor.start_monitoring(
            "int_001",
            intervention,
            baseline
        )

        self.assertIsNotNone(monitoring_id)
        self.assertTrue(monitoring_id.startswith("mon_"))
        self.assertEqual(len(self.monitor.monitoring_data), 1)

    def test_dummy_monitoring_data(self):
        """Test dummy monitoring data generation"""
        data = self.monitor.get_dummy_monitoring_data()

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

        for point in data:
            self.assertIsInstance(point, MonitoringDataPoint)
            self.assertGreaterEqual(point.productivity_score, 0)
            self.assertLessEqual(point.productivity_score, 10)

    def test_daily_monitoring(self):
        """Test daily monitoring check"""
        intervention = {
            "recommendation": "Async updates"
        }

        baseline = {
            "productivity_score": 6.0,
            "pr_review_time_hours": 48,
            "pr_merge_time_hours": 72,
            "blocked_prs": 12
        }

        self.monitor.start_monitoring("int_001", intervention, baseline)
        daily_check = self.monitor.monitor_daily()

        self.assertIn("days_elapsed", daily_check)
        self.assertIn("latest_metrics", daily_check)
        self.assertIn("improvements", daily_check)

    def test_side_effect_detection(self):
        """Test side effect detection"""
        intervention = {
            "recommendation": "Async updates"
        }

        baseline = {
            "productivity_score": 6.0,
            "pr_review_time_hours": 48,
            "pr_merge_time_hours": 72,
            "blocked_prs": 12
        }

        self.monitor.start_monitoring("int_001", intervention, baseline)
        self.monitor.monitor_daily()  # Get some data

        side_effects = self.monitor.detect_side_effects()

        self.assertIsInstance(side_effects, list)

    def test_rlhf_feedback_generation(self):
        """Test RLHF feedback generation"""
        feedback = self.monitor.generate_rlhf_feedback(
            "Async updates",
            "success"
        )

        self.assertIn("intervention", feedback)
        self.assertIn("outcome", feedback)
        self.assertIn("success", feedback)
        self.assertIn("learning", feedback)
        self.assertTrue(feedback["success"])

    def test_report_generation(self):
        """Test final report generation"""
        intervention = {
            "recommendation": "Async updates"
        }

        baseline = {
            "productivity_score": 6.0,
            "pr_review_time_hours": 48,
            "pr_merge_time_hours": 72,
            "blocked_prs": 12
        }

        self.monitor.start_monitoring("int_001", intervention, baseline)
        self.monitor.monitor_daily()

        report = self.monitor.generate_report()

        self.assertIn("status", report)
        self.assertIn("baseline", report)
        self.assertIn("final", report)
        self.assertIn("improvements", report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_anomaly_to_investigation_flow(self):
        """Test flow from anomaly detection to investigation start"""
        detector = AnomalyDetector()
        alert = detector.check_for_anomalies()

        self.assertIsNotNone(alert)
        self.assertIn("metrics", alert)

        # Verify alert has all required fields for investigator
        required_fields = ["timestamp", "severity", "metrics", "description"]
        for field in required_fields:
            self.assertIn(field, alert)

    def test_planner_receives_investigator_output(self):
        """Test that planner can work with investigator output"""
        investigator_output = "New daily 2-hour meetings started, blocking senior engineers from code reviews"

        planner = InterventionPlanner()
        similar_cases = planner.find_similar_cases(investigator_output)

        self.assertGreater(len(similar_cases), 0)

    def test_supervisor_receives_plan_output(self):
        """Test that supervisor can process planner output"""
        investigation = {
            "investigation_id": "inv_001",
            "confidence_score": 0.89,
            "root_cause": "Meeting overload",
            "evidence_trail": [{}, {}, {}]
        }

        plan = {
            "options": [{
                "cost": "low",
                "implementation_days": 1,
                "risk": "low"
            }],
            "recommendation": "Async updates"
        }

        supervisor = SupervisorAgent()
        result = supervisor.orchestrate(investigation, plan)

        self.assertIn("status", result)
        self.assertIn("execution", result)


def run_all_tests():
    """Run all tests with verbose output"""
    print("\n" + "=" * 80)
    print("🧪 RUNNING PRODUCTIVITY AGENT SYSTEM TEST SUITE")
    print("=" * 80 + "\n")

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestToolExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestRootCauseInvestigator))
    suite.addTests(loader.loadTestsFromTestCase(TestInterventionPlanner))
    suite.addTests(loader.loadTestsFromTestCase(TestSupervisorAgent))
    suite.addTests(loader.loadTestsFromTestCase(TestOutcomeMonitor))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)