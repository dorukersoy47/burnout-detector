# 🚀 Quick Start Cheat Sheet

## Installation (2 minutes)
```powershell
cd slack-integration
pip install -r requirements.txt
# Add your Slack token
```

## Get Slack Token
1. https://api.slack.com/apps → Create New App
2. OAuth & Permissions → Add scopes: `channels:history`, `channels:read`, `users:read`
3. Install to Workspace → Copy token (starts with `xoxb-`)
4. In Slack: `/invite @Slack Agent` in channels

## Run Agent
```powershell
python slack_agent.py
```

## Test Without Slack
```powershell
python test_agent.py
```

## View Events
```powershell
Get-ChildItem events
Get-Content events\slack_agent_mood_drop_*.json | ConvertFrom-Json
```

## Query Database
```powershell
sqlite3 slack_memory.db "SELECT sentiment_label, COUNT(*) FROM messages GROUP BY sentiment_label;"
```

## Key Files
- `slack_agent.py` - Main agent (500+ lines)
- `SETUP_GUIDE.md` - Full setup instructions
- `ARCHITECTURE.md` - Technical deep dive
- `event_integration_example.py` - How other agents connect

## Thresholds (in slack_agent.py)
```python
"negative_sentiment": -0.3      # More negative = lower
"sentiment_spike_delta": 0.4    # Change threshold
"message_burst_count": 10       # Messages in burst
"late_night_hour": 22           # 10 PM
```

## Event Types
- `mood_drop` - Avg sentiment too negative
- `sentiment_spike` - Rapid change
- `message_burst` - Too many messages fast
- `off_hours_activity` - Late night messaging

## Troubleshooting
| Error | Solution |
|-------|----------|
| Token not found | Check `.env` file exists |
| channel_not_found | Invite bot: `/invite @Burnout Detector` |
| No messages | Ensure channels have recent messages |
| Module not found | Run `pip install -r requirements.txt` |

## Next Steps
1. ✅ Set up Slack App (5 min)
2. ✅ Run test script
3. ✅ Send test messages in Slack
4. ✅ Check events generated
5. ✅ Build Calendar/Vision agents
6. ✅ Create Supervisor (LLM)

## Demo Script
1. Show autonomous monitoring
2. Send negative messages
3. Watch interval adjust
4. Query database
5. Show event bus

**Questions?** See `SETUP_GUIDE.md`
