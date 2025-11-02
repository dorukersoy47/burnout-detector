"""
Agentic Slack Agent for Burnout Detection
==========================================
This agent autonomously monitors Slack messages for burnout indicators.

Features:
- Sentiment analysis (positive/negative/neutral)
- Message frequency and timing patterns
- Response time monitoring
- Autonomous decision-making (re-analysis, threshold adjustment)
- Event publishing to shared event bus
- Memory persistence via SQLite
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import sqlite3
from collections import defaultdict

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SlackAgent:
    """
    Autonomous Slack monitoring agent with decision-making capabilities.
    """
    
    def __init__(self, slack_token: str, event_bus_path: str = "./events", 
                 db_path: str = "./slack_memory.db"):
        """
        Initialize the Slack Agent.
        
        Args:
            slack_token: Slack Bot Token
            event_bus_path: Path to event bus directory (JSON files)
            db_path: Path to SQLite database for memory
        """
        self.client = WebClient(token=slack_token)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.event_bus_path = Path(event_bus_path)
        self.event_bus_path.mkdir(exist_ok=True)
        self.db_path = db_path
        
        # Agent state
        self.agent_id = "slack_agent"
        self.objective = "Monitor Slack communication patterns to detect burnout indicators"
        self.is_running = False
        
        # Autonomous thresholds (can self-adjust)
        self.thresholds = {
            "negative_sentiment": -0.2,  # VADER compound score (lowered for better detection)
            "severe_negative_threshold": -0.4,  # Lowered to catch "burnt out" messages
            "sentiment_spike_delta": 0.3,  # Change in sentiment (more sensitive)
            "message_burst_count": 10,  # Messages in short time
            "message_burst_window": 300,  # 5 minutes in seconds
            "late_night_hour": 22,  # 10 PM
            "early_morning_hour": 6,  # 6 AM
            "high_response_time": 3600,  # 1 hour in seconds
        }
        
        # Memory for autonomous decisions
        self.recent_observations = []
        self.last_analysis_time = None
        self.analysis_interval = 300  # 5 minutes default
        
        # Initialize database
        self._init_database()
        
        print(f"[{self.agent_id}] Initialized with objective: {self.objective}")
    
    def _init_database(self):
        """Initialize SQLite database for memory persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                channel_id TEXT,
                text TEXT,
                timestamp REAL,
                sentiment_score REAL,
                sentiment_label TEXT,
                analyzed_at REAL
            )
        """)
        
        # Observations table (what agent "observes")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                observation_type TEXT,
                data TEXT,
                confidence REAL,
                triggered_action TEXT
            )
        """)
        
        # Agent state table (for reflection)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_state (
                timestamp REAL,
                threshold_name TEXT,
                threshold_value REAL,
                reason TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"[{self.agent_id}] Database initialized at {self.db_path}")
    
    def _publish_event(self, event_type: str, sentiment_score: float, messages: List[Dict]):
        """
        Publish event to the event bus (JSON file).
        
        Args:
            event_type: Type of pattern detected
            sentiment_score: Overall sentiment score (-1.0 to +1.0)
            messages: List of relevant messages (up to 10 most relevant)
        """
        # Select up to 10 most relevant messages (prioritize negative ones)
        sorted_messages = sorted(messages, key=lambda m: m.get('sentiment_score', 0))
        top_messages = sorted_messages[:10]
        
        event = {
            "agent_id": self.agent_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "sentiment_score": round(sentiment_score, 3),
            "messages": [
                {
                    "text": msg['text'],
                    "sentiment": round(msg['sentiment_score'], 3),
                    "user_id": msg.get('user_id', 'unknown'),
                    "timestamp": msg['timestamp']
                }
                for msg in top_messages
            ]
        }
        
        # Write to event bus
        event_file = self.event_bus_path / f"{self.agent_id}_{event_type}_{int(time.time())}.json"
        with open(event_file, 'w') as f:
            json.dump(event, f, indent=2)
        
        print(f"[{self.agent_id}] Published event: {event_type} (sentiment: {sentiment_score:.2f})")
    
    def _store_observation(self, obs_type: str, data: Dict, sentiment_score: float, action: str = None):
        """Store observation in memory for reflection."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO observations (timestamp, observation_type, data, confidence, triggered_action)
            VALUES (?, ?, ?, ?, ?)
        """, (time.time(), obs_type, json.dumps(data), sentiment_score, action))
        conn.commit()
        conn.close()
    
    def _store_message(self, msg_id: str, user_id: str, channel_id: str, 
                       text: str, timestamp: float, sentiment_score: float, sentiment_label: str):
        """Store analyzed message in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO messages 
            (id, user_id, channel_id, text, timestamp, sentiment_score, sentiment_label, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (msg_id, user_id, channel_id, text, timestamp, sentiment_score, sentiment_label, time.time()))
        conn.commit()
        conn.close()
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using VADER.
        
        Returns:
            Dict with 'compound' score and 'label' (positive/negative/neutral)
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return {
            "compound": compound,
            "label": label,
            "scores": scores
        }
    
    def _get_recent_messages(self, hours: int = 24) -> List[Dict]:
        """Retrieve recent messages from memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_time = time.time() - (hours * 3600)
        cursor.execute("""
            SELECT * FROM messages WHERE timestamp > ? ORDER BY timestamp DESC
        """, (cutoff_time,))
        
        columns = [desc[0] for desc in cursor.description]
        messages = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return messages
    
    async def _fetch_slack_messages(self, hours: int = 1) -> List[Dict]:
        """
        Fetch messages from Slack (all public channels + DMs if permissions allow).
        
        Args:
            hours: How many hours back to fetch
        """
        messages = []
        oldest_timestamp = (datetime.now() - timedelta(hours=hours)).timestamp()
        
        try:
            # Get all conversations (channels + DMs)
            result = self.client.conversations_list(types="public_channel,private_channel,im,mpim")
            channels = result['channels']
            
            for channel in channels:
                try:
                    # Fetch messages from each channel
                    history = self.client.conversations_history(
                        channel=channel['id'],
                        oldest=str(oldest_timestamp)
                    )
                    
                    for message in history.get('messages', []):
                        if message.get('type') == 'message' and 'text' in message:
                            text = message['text']
                            
                            # Skip system messages and bot messages
                            if message.get('subtype') in ['channel_join', 'channel_leave', 'bot_message']:
                                continue
                            if '<@' in text and 'has joined the channel' in text:
                                continue
                            if text.strip() == '':
                                continue
                            
                            messages.append({
                                'id': f"{channel['id']}_{message['ts']}",
                                'user_id': message.get('user', 'unknown'),
                                'channel_id': channel['id'],
                                'text': text,
                                'timestamp': float(message['ts'])
                            })
                except SlackApiError as e:
                    if e.response['error'] != 'channel_not_found':
                        print(f"[{self.agent_id}] Warning: Could not fetch from channel {channel.get('name', channel['id'])}: {e}")
        
        except SlackApiError as e:
            print(f"[{self.agent_id}] Error fetching conversations: {e}")
        
        return messages
    
    def _detect_sentiment_patterns(self, messages: List[Dict]) -> List[Dict]:
        """
        Detect sentiment-based burnout patterns.
        
        Returns list of findings with:
        - type: Pattern type
        - sentiment_score: Overall sentiment (-1.0 to +1.0)
        - messages: Relevant messages for this pattern
        """
        findings = []
        
        if len(messages) == 0:
            return findings
        
        # 1. Check for severe individual negative messages (lowered threshold to -0.4)
        severe_messages = [msg for msg in messages 
                          if msg['sentiment_score'] < self.thresholds.get('severe_negative_threshold', -0.4)]
        if severe_messages:
            avg_sentiment = sum(m['sentiment_score'] for m in severe_messages) / len(severe_messages)
            findings.append({
                "type": "severe_negative_message",
                "sentiment_score": avg_sentiment,
                "messages": severe_messages
            })
        
        if len(messages) < 2:
            return findings
        
        # 2. Check for overall mood drop (average sentiment negative) - COMMENTED OUT
        # sentiments = [msg['sentiment_score'] for msg in messages]
        # avg_sentiment = sum(sentiments) / len(sentiments)
        # 
        # if avg_sentiment < self.thresholds['negative_sentiment']:
        #     findings.append({
        #         "type": "mood_drop",
        #         "sentiment_score": avg_sentiment,
        #         "messages": messages
        #     })
        
        # 3. Check for sentiment spike (emotional volatility) - COMMENTED OUT
        # if len(messages) >= 5:
        #     sentiments = [msg['sentiment_score'] for msg in messages]
        #     recent_avg = sum(sentiments[:5]) / 5
        #     older_avg = sum(sentiments[5:min(15, len(sentiments))]) / len(sentiments[5:min(15, len(sentiments))])
        #     delta = abs(recent_avg - older_avg)
        #     
        #     if delta > self.thresholds['sentiment_spike_delta']:
        #         findings.append({
        #             "type": "sentiment_spike",
        #             "sentiment_score": recent_avg,
        #             "messages": messages[:5]  # Most recent messages showing the spike
        #         })
        
        return findings
    
    def _detect_timing_patterns(self, messages: List[Dict]) -> List[Dict]:
        """
        Detect timing-based burnout patterns (off-hours work, message bursts).
        
        Returns list of findings with sentiment and messages.
        """
        findings = []
        
        if not messages:
            return findings
        
        # 1. Off-hours activity (late night / early morning)
        off_hours_messages = []
        
        for msg in messages:
            dt = datetime.fromtimestamp(msg['timestamp'])
            hour = dt.hour
            
            if hour >= self.thresholds['late_night_hour'] or hour <= self.thresholds['early_morning_hour']:
                off_hours_messages.append(msg)
        
        if off_hours_messages and len(off_hours_messages) / len(messages) > 0.3:
            avg_sentiment = sum(m['sentiment_score'] for m in off_hours_messages) / len(off_hours_messages)
            findings.append({
                "type": "off_hours_activity",
                "sentiment_score": avg_sentiment,
                "messages": off_hours_messages
            })
        
        # 2. Message bursts (posting too frequently)
        if len(messages) >= self.thresholds['message_burst_count']:
            timestamps = sorted([msg['timestamp'] for msg in messages])
            
            for i in range(len(timestamps) - self.thresholds['message_burst_count'] + 1):
                window_start = timestamps[i]
                window_end = timestamps[i + self.thresholds['message_burst_count'] - 1]
                
                if (window_end - window_start) < self.thresholds['message_burst_window']:
                    # Get messages in this burst window
                    burst_messages = [m for m in messages 
                                    if window_start <= m['timestamp'] <= window_end]
                    avg_sentiment = sum(m['sentiment_score'] for m in burst_messages) / len(burst_messages)
                    
                    findings.append({
                        "type": "message_burst",
                        "sentiment_score": avg_sentiment,
                        "messages": burst_messages
                    })
                    break  # Only report once
        
        return findings
    
    async def _autonomous_analysis_cycle(self):
        """
        Main autonomous loop: decide when and what to analyze.
        """
        print(f"[{self.agent_id}] Starting autonomous analysis cycle...")
        
        # Fetch new messages
        new_messages = await self._fetch_slack_messages(hours=1)
        print(f"[{self.agent_id}] Fetched {len(new_messages)} messages")
        
        # Analyze sentiment for each message
        analyzed_messages = []
        for msg in new_messages:
            sentiment = self._analyze_sentiment(msg['text'])
            msg['sentiment_score'] = sentiment['compound']
            msg['sentiment_label'] = sentiment['label']
            analyzed_messages.append(msg)
            
            # Store in memory
            self._store_message(
                msg['id'], msg['user_id'], msg['channel_id'],
                msg['text'], msg['timestamp'],
                sentiment['compound'], sentiment['label']
            )
        
        # Detect patterns (autonomous reasoning)
        sentiment_findings = self._detect_sentiment_patterns(analyzed_messages)
        timing_findings = self._detect_timing_patterns(analyzed_messages)
        
        all_findings = sentiment_findings + timing_findings
        
        # Publish events for all findings
        for finding in all_findings:
            self._publish_event(
                event_type=finding['type'],
                sentiment_score=finding['sentiment_score'],
                messages=finding['messages']
            )
            self._store_observation(
                obs_type=finding['type'],
                data=finding,
                sentiment_score=finding['sentiment_score'],
                action="published_event"
            )
        
        # AUTONOMOUS DECISION: Should we re-analyze sooner?
        # High severity if sentiment < -0.5 or many findings
        high_severity = any(f['sentiment_score'] < -0.5 for f in all_findings) or len(all_findings) >= 2
        if high_severity:
            # High-confidence finding → reduce analysis interval (be more vigilant)
            old_interval = self.analysis_interval
            self.analysis_interval = max(60, self.analysis_interval // 2)  # At least 1 minute
            print(f"[{self.agent_id}] 🤖 AUTONOMOUS DECISION: High confidence finding detected. "
                  f"Reducing analysis interval from {old_interval}s to {self.analysis_interval}s")
            
            # Log decision
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO agent_state (timestamp, threshold_name, threshold_value, reason)
                VALUES (?, ?, ?, ?)
            """, (time.time(), "analysis_interval", self.analysis_interval, 
                  "High-confidence burnout indicator detected"))
            conn.commit()
            conn.close()
        else:
            # No concerning findings → gradually return to normal interval
            if self.analysis_interval < 300:
                self.analysis_interval = min(300, self.analysis_interval + 30)
                print(f"[{self.agent_id}] 🤖 AUTONOMOUS DECISION: No concerning patterns. "
                      f"Increasing interval to {self.analysis_interval}s")
        
        self.last_analysis_time = time.time()
    
    async def run(self):
        """
        Main agent loop - runs continuously with autonomous decision-making.
        """
        self.is_running = True
        print(f"\n{'='*60}")
        print(f"[{self.agent_id}] 🚀 STARTING AGENTIC OPERATION")
        print(f"Objective: {self.objective}")
        print(f"{'='*60}\n")
        
        while self.is_running:
            try:
                await self._autonomous_analysis_cycle()
                
                # Wait for next cycle (interval determined autonomously)
                print(f"[{self.agent_id}] Sleeping for {self.analysis_interval}s until next analysis...\n")
                await asyncio.sleep(self.analysis_interval)
                
            except KeyboardInterrupt:
                print(f"\n[{self.agent_id}] Received shutdown signal")
                break
            except Exception as e:
                print(f"[{self.agent_id}] Error in analysis cycle: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying
    
    def stop(self):
        """Stop the agent."""
        self.is_running = False
        print(f"[{self.agent_id}] Stopping...")
    
    def get_memory_summary(self, hours: int = 24) -> Dict:
        """
        Get a summary of recent observations and patterns (for reflection).
        """
        messages = self._get_recent_messages(hours)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_time = time.time() - (hours * 3600)
        
        cursor.execute("""
            SELECT observation_type, COUNT(*), AVG(confidence)
            FROM observations
            WHERE timestamp > ?
            GROUP BY observation_type
        """, (cutoff_time,))
        
        observations = cursor.fetchall()
        conn.close()
        
        return {
            "total_messages": len(messages),
            "avg_sentiment": sum(m['sentiment_score'] for m in messages) / len(messages) if messages else 0,
            "observations": [
                {"type": obs[0], "count": obs[1], "avg_sentiment": obs[2]}
                for obs in observations
            ],
            "current_analysis_interval": self.analysis_interval
        }
    
    def query(self, 
              hours: int = 24, 
              min_sentiment: float = None,
              max_sentiment: float = None,
              pattern_type: str = None,
              limit: int = 100) -> Dict:
        """
        Query messages and observations from memory.
        
        Args:
            hours: Time window in hours (default 24)
            min_sentiment: Filter messages with sentiment >= this value
            max_sentiment: Filter messages with sentiment <= this value
            pattern_type: Filter by specific pattern type
            limit: Maximum number of results
        
        Returns:
            Dict with messages and observations
        
        Example:
            # Get all negative messages from last 12 hours
            agent.query(hours=12, max_sentiment=-0.2)
            
            # Get severe burnout indicators
            agent.query(pattern_type='severe_negative_message')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff_time = time.time() - (hours * 3600)
        
        # Build message query
        msg_query = "SELECT * FROM messages WHERE timestamp > ?"
        msg_params = [cutoff_time]
        
        if min_sentiment is not None:
            msg_query += " AND sentiment_score >= ?"
            msg_params.append(min_sentiment)
        
        if max_sentiment is not None:
            msg_query += " AND sentiment_score <= ?"
            msg_params.append(max_sentiment)
        
        msg_query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(msg_query, msg_params)
        columns = [desc[0] for desc in cursor.description]
        messages = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Build observations query
        obs_query = "SELECT * FROM observations WHERE timestamp > ?"
        obs_params = [cutoff_time]
        
        if pattern_type:
            obs_query += " AND observation_type = ?"
            obs_params.append(pattern_type)
        
        obs_query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        cursor.execute(obs_query, obs_params)
        columns = [desc[0] for desc in cursor.description]
        observations = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "query": {
                "hours": hours,
                "min_sentiment": min_sentiment,
                "max_sentiment": max_sentiment,
                "pattern_type": pattern_type,
                "limit": limit
            },
            "messages": messages,
            "observations": observations,
            "summary": {
                "total_messages": len(messages),
                "total_observations": len(observations),
                "avg_sentiment": sum(m['sentiment_score'] for m in messages) / len(messages) if messages else None
            }
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get Slack token from environment
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    
    if not slack_token:
        print("ERROR: SLACK_BOT_TOKEN not found in environment!")
        print("Please create a .env file with your Slack bot token.")
        exit(1)
    
    # Create and run agent
    agent = SlackAgent(
        slack_token=slack_token,
        event_bus_path="./events",
        db_path="./slack_memory.db"
    )
    
    # Run the agent
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        agent.stop()
