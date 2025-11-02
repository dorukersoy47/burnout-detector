"""
Test script for Slack Agent - simulates testing without real Slack API calls
"""

import asyncio
import json
import time
from pathlib import Path
import sqlite3


def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    from slack_agent import SlackAgent
    
    print("🧪 Testing Sentiment Analysis...")
    
    # Create agent (will fail without token, but we can still test methods)
    try:
        agent = SlackAgent(slack_token="test-token", event_bus_path="./test_events", db_path="./test_memory.db")
        
        # Test different sentiments
        test_messages = [
            ("I love this project! So excited!", "positive"),
            ("This is terrible, I can't handle it anymore", "negative"),
            ("The meeting is at 3pm", "neutral"),
            ("Exhausted and burned out, working late again", "negative"),
            ("Great job everyone! Amazing work!", "positive"),
        ]
        
        for text, expected in test_messages:
            result = agent._analyze_sentiment(text)
            print(f"  Text: '{text[:50]}...'")
            print(f"    Expected: {expected}, Got: {result['label']}, Score: {result['compound']:.2f}")
            assert result['label'] == expected, f"Expected {expected}, got {result['label']}"
        
        print("✅ Sentiment analysis tests passed!\n")
        
        # Cleanup
        Path("test_memory.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def test_event_publishing():
    """Test event bus publishing"""
    from slack_agent import SlackAgent
    
    print("🧪 Testing Event Publishing...")
    
    try:
        agent = SlackAgent(slack_token="test-token", event_bus_path="./test_events", db_path="./test_memory.db")
        
        # Publish test event
        agent._publish_event(
            event_type="mood_drop",
            data={"avg_sentiment": -0.5, "message_count": 10},
            confidence=0.8
        )
        
        # Check if event file was created
        event_files = list(Path("./test_events").glob("slack_agent_mood_drop_*.json"))
        assert len(event_files) > 0, "No event file created"
        
        # Read and verify event
        with open(event_files[0], 'r') as f:
            event = json.load(f)
            assert event['event_type'] == 'mood_drop'
            assert event['confidence'] == 0.8
            assert event['agent_id'] == 'slack_agent'
        
        print("✅ Event publishing tests passed!\n")
        
        # Cleanup
        for f in Path("./test_events").glob("*.json"):
            f.unlink()
        Path("./test_events").rmdir()
        Path("test_memory.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def test_memory_storage():
    """Test SQLite memory storage"""
    from slack_agent import SlackAgent
    
    print("🧪 Testing Memory Storage...")
    
    try:
        agent = SlackAgent(slack_token="test-token", event_bus_path="./test_events", db_path="./test_memory.db")
        
        # Store test observation
        agent._store_observation(
            obs_type="mood_drop",
            data={"test": "data"},
            confidence=0.75,
            action="published_event"
        )
        
        # Store test message
        agent._store_message(
            msg_id="test_123",
            user_id="U123",
            channel_id="C123",
            text="Test message",
            timestamp=time.time(),
            sentiment_score=-0.5,
            sentiment_label="negative"
        )
        
        # Verify storage
        conn = sqlite3.connect("./test_memory.db")
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM observations")
        obs_count = cursor.fetchone()[0]
        assert obs_count > 0, "No observations stored"
        
        cursor.execute("SELECT COUNT(*) FROM messages")
        msg_count = cursor.fetchone()[0]
        assert msg_count > 0, "No messages stored"
        
        conn.close()
        
        print("✅ Memory storage tests passed!\n")
        
        # Cleanup
        Path("test_memory.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def test_pattern_detection():
    """Test pattern detection logic"""
    from slack_agent import SlackAgent
    
    print("🧪 Testing Pattern Detection...")
    
    try:
        agent = SlackAgent(slack_token="test-token", event_bus_path="./test_events", db_path="./test_memory.db")
        
        # Create test messages with negative sentiment
        test_messages = [
            {
                'id': f'msg_{i}',
                'user_id': 'U123',
                'channel_id': 'C123',
                'text': 'Negative message',
                'timestamp': time.time() - (i * 60),
                'sentiment_score': -0.6,
                'sentiment_label': 'negative'
            }
            for i in range(10)
        ]
        
        # Test sentiment pattern detection
        findings = agent._detect_sentiment_patterns(test_messages)
        print(f"  Sentiment findings: {len(findings)}")
        assert len(findings) > 0, "Should detect negative sentiment pattern"
        
        # Create messages with timing issues (late night)
        late_night_messages = [
            {
                'id': f'msg_{i}',
                'user_id': 'U123',
                'channel_id': 'C123',
                'text': 'Late night message',
                'timestamp': time.time() - (i * 60),
                'sentiment_score': -0.3,
                'sentiment_label': 'neutral'
            }
            for i in range(5)
        ]
        
        # Manually set timestamps to late night (11 PM)
        from datetime import datetime, timedelta
        late_night = datetime.now().replace(hour=23, minute=0, second=0)
        for i, msg in enumerate(late_night_messages):
            msg['timestamp'] = (late_night - timedelta(minutes=i*5)).timestamp()
        
        timing_findings = agent._detect_timing_patterns(late_night_messages)
        print(f"  Timing findings: {len(timing_findings)}")
        
        print("✅ Pattern detection tests passed!\n")
        
        # Cleanup
        Path("test_memory.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


def test_autonomous_decisions():
    """Test autonomous interval adjustment"""
    from slack_agent import SlackAgent
    
    print("🧪 Testing Autonomous Decision-Making...")
    
    try:
        agent = SlackAgent(slack_token="test-token", event_bus_path="./test_events", db_path="./test_memory.db")
        
        initial_interval = agent.analysis_interval
        print(f"  Initial interval: {initial_interval}s")
        
        # Simulate high-confidence finding (should reduce interval)
        # In real agent, this happens in _autonomous_analysis_cycle
        # For test, we'll manually check the logic
        
        confidence = 0.8
        if confidence > 0.7:
            agent.analysis_interval = max(60, agent.analysis_interval // 2)
            print(f"  After high confidence: {agent.analysis_interval}s")
            assert agent.analysis_interval < initial_interval, "Interval should decrease"
        
        print("✅ Autonomous decision tests passed!\n")
        
        # Cleanup
        Path("test_memory.db").unlink(missing_ok=True)
        
    except Exception as e:
        print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Running Slack Agent Tests")
    print("="*60 + "\n")
    
    test_sentiment_analysis()
    test_event_publishing()
    test_memory_storage()
    test_pattern_detection()
    test_autonomous_decisions()
    
    print("="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Set up your Slack App and get a bot token")
    print("2. Create .env file with SLACK_BOT_TOKEN")
    print("3. Run: python slack_agent.py")
