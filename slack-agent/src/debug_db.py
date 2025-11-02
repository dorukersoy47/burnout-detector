import sqlite3
import time
from datetime import datetime

conn = sqlite3.connect('slack_memory.db')
cursor = conn.cursor()

print("\n=== ALL MESSAGES ===")
cursor.execute('SELECT text, sentiment_score, sentiment_label, timestamp FROM messages ORDER BY timestamp DESC')
messages = cursor.fetchall()

now = time.time()
for msg in messages:
    text, score, label, ts = msg
    age_minutes = (now - ts) / 60
    dt = datetime.fromtimestamp(ts)
    print(f"\n{dt.strftime('%Y-%m-%d %H:%M:%S')} ({age_minutes:.0f}m ago)")
    print(f"  Text: {text[:80]}")
    print(f"  Score: {score:.2f} | Label: {label}")

print(f"\n=== STATISTICS ===")
print(f"Total messages: {len(messages)}")

# Check last hour
cursor.execute('SELECT COUNT(*), AVG(sentiment_score) FROM messages WHERE timestamp > ?', (now - 3600,))
recent = cursor.fetchone()
print(f"Messages in last hour: {recent[0]}")
if recent[1] is not None:
    print(f"Average sentiment (last hour): {recent[1]:.2f}")
    print(f"Threshold for mood_drop: -0.3")
    if recent[1] < -0.3:
        print(f"✓ SHOULD TRIGGER mood_drop event!")
    else:
        print(f"✗ Not negative enough for mood_drop")

print(f"\n=== OBSERVATIONS ===")
cursor.execute('SELECT COUNT(*) FROM observations')
obs_count = cursor.fetchone()[0]
print(f"Total observations: {obs_count}")

if obs_count > 0:
    cursor.execute('SELECT observation_type, confidence, timestamp FROM observations ORDER BY timestamp DESC LIMIT 5')
    for obs in cursor.fetchall():
        dt = datetime.fromtimestamp(obs[2])
        print(f"  {obs[0]} | Confidence: {obs[1]:.2f} | {dt.strftime('%H:%M:%S')}")

conn.close()
