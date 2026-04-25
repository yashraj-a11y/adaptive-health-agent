# Adaptive Personal Health Agent
## Complete Planning & Architecture Document
### Agentic AI Hackathon — IoT Wearable Context Engine Track

---

> **The Product In One Sentence:**
> A personal health agent that builds a deep, evolving model of you over time — acting like the world's most attentive health advisor who never forgets anything, notices patterns you cannot see yourself, understands your trajectory not just your current state, and knows when to speak and when to stay quiet.

---

# TABLE OF CONTENTS

1. [Product Vision](#1-product-vision)
2. [What Makes This Different](#2-what-makes-this-different)
3. [Tech Stack](#3-tech-stack)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Mock Telemetry JSON Stream](#5-mock-telemetry-json-stream)
6. [Memory Architecture](#6-memory-architecture)
7. [Onboarding Flow](#7-onboarding-flow)
8. [Agent 1 — The Profiler](#8-agent-1--the-profiler)
9. [Agent 2 — The Analyst](#9-agent-2--the-analyst)
10. [Agent 3 — The Communicator](#10-agent-3--the-communicator)
11. [Interaction Modes](#11-interaction-modes)
12. [Medical Knowledge Base](#12-medical-knowledge-base)
13. [Complete System Flow](#13-complete-system-flow)
14. [LangGraph Implementation Guide](#14-langgraph-implementation-guide)
15. [Folder Structure](#15-folder-structure)
16. [Build Order](#16-build-order)

---

# 1. Product Vision

## The Problem With Every Health App Today

Every health app and smartwatch today — including Apple Watch — is **reactive and stateless.**

- It wakes up every day knowing nothing about you as a *person*
- It knows your age and gender from your profile setup. That is all.
- It fires the same alert for the same number regardless of who you are
- Heart rate 110? Alert. For every single user. Whether they are an athlete, an anxious student, or a 65-year-old with hypertension.
- It answers only one question: *"Is something wrong right now?"*

## What This Product Does Instead

This agent answers something fundamentally deeper:

> **"Who is this person, how are they actually doing across time, and what do they need to hear — that they haven't realized themselves yet?"**

It builds a **living health fingerprint** of the user. Not a snapshot — a trajectory. Not a population comparison — a personal baseline. Not a threshold alert — a pattern recognition system that knows *this specific person.*

## The Core Differentiators

**1. Anomaly detection relative to personal baseline — not population average**
Not "HR 95 is elevated." But "HR 95 at 11pm while sitting is 34% above YOUR personal evening baseline — combined with 3 nights of poor sleep, this suggests accumulated stress, not a one-time event."

**2. Multi-signal context fusion**
No single alert fires on one metric. The agent looks at combinations. HR spike + complete stillness + 2am = very different from HR spike + walking + afternoon.

**3. Longitudinal pattern recognition**
"You have had 4 consecutive nights under 6 hours sleep. Your cognitive performance window tomorrow is likely compromised."

**4. The unsaid intervention**
The agent decides *not* to tell you something — because it knows you. It holds back minor anomalies and only surfaces things when they form a real pattern.

**5. Proactive questioning**
Like a doctor who asks "is this normal for you?" — the agent asks contextual questions when it encounters uncertainty, building a richer model over time.

**6. Adaptive communication**
The agent learns how *this person* likes to be spoken to. Direct and clinical, or warm and conversational. It adapts over time based on engagement signals.

**7. User-initiated conversation**
The user can talk to the agent anytime — asking about their health, requesting explanations, checking in. Every answer is personalized to their specific history and data.

**8. Family and emergency awareness**
For vulnerable users, the agent has a wider circle of responsibility. It knows when to notify emergency contacts. Like a doctor who talks to the family, not just the patient.

---

# 2. What Makes This Different

## What Apple Watch Already Does
- Threshold-based alerts (HR too high → alert)
- Basic context awareness (elevated HR while inactive → alert)
- Irregular rhythm detection
- Low SpO2 alert

## What This Agent Does That No App Does Today

| Capability | Apple Watch | This Agent |
|---|---|---|
| Personal baseline learning | ❌ | ✅ |
| Multi-week pattern recognition | ❌ | ✅ |
| Multi-signal context fusion | Limited | ✅ Deep |
| Trajectory intelligence | ❌ | ✅ |
| Proactive questioning | ❌ | ✅ |
| User-initiated conversation | ❌ | ✅ |
| Communication style adaptation | ❌ | ✅ |
| "You're lying to yourself" insights | ❌ | ✅ |
| Family notification with context | ❌ | ✅ |
| Knows when to stay quiet | ❌ | ✅ |
| RAG on personal health history | ❌ | ✅ |

---

# 3. Tech Stack

## Overview

| Layer | Tool | Why |
|---|---|---|
| LLM | Groq API | Fast inference, generous free tier, multiple API keys for different agents |
| Agent Orchestration | LangGraph | Graph-based multi-agent flow with state management, perfect for this architecture |
| Vector Database | ChromaDB | Free, runs locally, no cloud account needed, semantic retrieval |
| Telemetry Simulation | Python script | Custom JSON stream simulating smartwatch data |
| UI | Streamlit | Fast to build, Python-based, clean demo interface |
| Notifications | Console output | Simulated family/emergency alerts for hackathon |
| Language | Python | Primary throughout |

## Groq API Notes

- Use **separate API keys for each agent** — Profiler, Analyst, Communicator each get their own key
- This makes the system feel genuinely distributed and is more robust
- Model recommendation: `llama3-70b-8192` or `mixtral-8x7b-32768` — both fast and capable on Groq free tier
- Keep system prompts lean to preserve context window for memory and telemetry

## LangGraph Notes

- Agents are defined as **nodes** in a graph
- Edges define flow between agents with conditional routing
- State is maintained across the entire graph — Living Profile lives here
- Supports cycles — Analyst can loop back to Profiler if it needs more data
- Install: `pip install langgraph langchain`

## ChromaDB Notes

- Runs entirely locally — no API key, no cloud account
- Two collections needed:
  - `episodic_memory` — health events, anomalies, conversations
  - `medical_knowledge_base` — clinical knowledge documents
- Install: `pip install chromadb`
- Embedding model: use `all-MiniLM-L6-v2` from sentence-transformers (free, fast, local)

## Streamlit Notes

- Single Python file for the entire UI
- Shows: incoming telemetry stream, agent activity, conversation interface, Living Profile summary
- Install: `pip install streamlit`
- Run: `streamlit run app.py`

---

# 4. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TELEMETRY STREAM                          │
│              (Mock JSON — Python script)                     │
│         Sends one packet every N seconds                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT 1: PROFILER                          │
│                                                              │
│  • Receives every JSON packet                                │
│  • Compares to Living Profile baselines                      │
│  • Updates rolling state                                     │
│  • Logs significant events to ChromaDB                       │
│  • Detects confirmed patterns (3+ packets)                   │
│  • Runs 7-day summarization pass                             │
│  • Activates Analyst when needed                             │
│  • Can initiate questions to user                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
┌──────────────────┐         ┌───────────────────────┐
│   CHROMADB       │         │   LIVING PROFILE       │
│                  │         │   (LangGraph State)    │
│ • Episodic       │◄────────│                        │
│   Memory         │         │ • Always in context    │
│ • Medical        │────────►│ • ~400 tokens          │
│   Knowledge      │         │ • Updated every 7 days │
│   Base           │         │                        │
└──────────────────┘         └───────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                   AGENT 2: ANALYST                           │
│                                                              │
│  • Reads Living Profile (always present)                     │
│  • RAG query to ChromaDB episodic memory                     │
│  • RAG query to medical knowledge base                       │
│  • Runs decision framework                                   │
│  • Classifies severity Level 1-5                             │
│  • Checks timing and context                                 │
│  • Drafts content substance                                  │
│  • Passes to Communicator                                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                AGENT 3: COMMUNICATOR                         │
│                                                              │
│  • Reads communication profile from Living Profile           │
│  • Adapts tone, style, length, framing                       │
│  • Delivers message or question to user                      │
│  • Logs interaction to ChromaDB                              │
│  • Updates communication profile from response signals       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      USER                                    │
│                                                              │
│  • Receives agent messages                                   │
│  • Can respond to agent questions                            │
│  • Can initiate conversation anytime                         │
│  • Emergency contact notified if Level 5                     │
└─────────────────────────────────────────────────────────────┘
```

---

# 5. Mock Telemetry JSON Stream

## Design Philosophy

The mock JSON stream is not random numbers. It is a **carefully designed narrative simulation** representing a real person with real patterns. Every field serves a purpose in powering the personalization system.

The stream contains only **raw sensor data and immediate context.** Nothing pre-computed. The system earns every insight it produces by computing trends, averages, and patterns itself from stored history in ChromaDB.

## Packet Structure

```json
{
  "timestamp": "2025-04-25T23:14:00",
  "user_id": "user_a",

  "vitals": {
    "heart_rate": 92,
    "hrv": 28,
    "spo2": 97,
    "skin_temperature": 36.8,
    "breathing_rate": 17,
    "stress_score": 72,
    "recovery_score": 34,
    "eda_stress_indicator": "elevated"
  },

  "movement": {
    "steps_today": 2340,
    "activity_state": "sedentary",
    "activity_intensity": 0.1,
    "calories_burned": 1240,
    "active_minutes_today": 18
  },

  "sleep_last_night": {
    "total_hours": 4.8,
    "deep_sleep_percentage": 14,
    "rem_percentage": 18,
    "light_sleep_percentage": 68,
    "sleep_efficiency": 61,
    "woke_up_times": 3,
    "sleep_onset_minutes": 35
  },

  "context": {
    "time_of_day": "night",
    "day_of_week": "Thursday",
    "is_weekend": false,
    "battery_level": 73,
    "location_zone": "home",
    "weather_temp_celsius": 31
  },

  "user_reported": {
    "mood": null,
    "stress_level": null,
    "notes": null
  }
}
```

## Field Explanations

### Vitals

| Field | Description | Normal Range | Notes |
|---|---|---|---|
| `heart_rate` | BPM current reading | 60-100 resting | Personal baseline matters more than population range |
| `hrv` | Heart Rate Variability in ms | 20-70ms (varies widely by person) | Lower = more stressed / less recovered |
| `spo2` | Blood oxygen percentage | 95-100% | Below 94% warrants attention |
| `skin_temperature` | Celsius | 36.0-37.2°C | Elevated = possible illness onset |
| `breathing_rate` | Breaths per minute | 12-20 | Elevated at rest = stress or illness |
| `stress_score` | Computed 0-100 | 0-25 = low, 25-50 = normal, 50-75 = elevated, 75+ = high | Derived from HRV patterns |
| `recovery_score` | Computed 0-100 | Above 70 = well recovered | Composite of sleep, HRV, activity |
| `eda_stress_indicator` | Electrodermal activity | `low` / `normal` / `elevated` / `high` | Emotional arousal proxy |

### Movement

| Field | Description | Notes |
|---|---|---|
| `steps_today` | Cumulative steps since midnight | Resets at midnight |
| `activity_state` | Current activity | `sedentary` / `light` / `moderate` / `intense` / `sleeping` |
| `activity_intensity` | 0.0 to 1.0 scale | 0 = completely still, 1 = maximum intensity |
| `calories_burned` | Cumulative today | Active + resting calories |
| `active_minutes_today` | Minutes of moderate+ activity | WHO recommends 30/day |

### Sleep

> **Note on sleep metrics:** Consumer wearables infer sleep stages from HRV and accelerometer data — not EEG. These are estimates, not clinical measurements. This is standard practice for all major consumer wearables (Fitbit, Garmin, Apple Watch). Reporting as percentages is more honest than exact hours for inferred data.

| Field | Description | Healthy Target |
|---|---|---|
| `total_hours` | Total sleep duration | 7-9 hours for adults |
| `deep_sleep_percentage` | % of sleep in deep/slow-wave stage | 15-25% |
| `rem_percentage` | % of sleep in REM stage | 20-25% |
| `light_sleep_percentage` | % of sleep in light stage | 50-60% |
| `sleep_efficiency` | % of time in bed actually sleeping | Above 85% |
| `woke_up_times` | Number of awakenings | 0-2 normal |
| `sleep_onset_minutes` | Time to fall asleep | Under 20 mins ideal |

### Context

| Field | Description | Notes |
|---|---|---|
| `time_of_day` | `morning` / `afternoon` / `evening` / `night` | Affects interpretation of all vitals |
| `day_of_week` | Full day name | Enables day-of-week pattern detection |
| `is_weekend` | Boolean | Behavioral patterns differ weekday vs weekend |
| `battery_level` | Watch battery % | Affects agent interruption decisions |
| `location_zone` | `home` / `work` / `commute` / `gym` / `unknown` | Context for activity and stress interpretation |
| `weather_temp_celsius` | Ambient temperature | Heat affects HR and sleep quality |

### User Reported

These fields are `null` by default. They get populated when:
- The agent asks the user a question and the user responds
- The user initiates a conversation and provides context
- The user voluntarily logs a mood or note

This closes the loop between conversation and telemetry data.

## Simulation Script Design

The mock script should tell a story — not emit random values. Design two simulation scenarios:

### Scenario A — Stress Accumulation Arc
```
Packets 1-10:   Normal baseline. Everything calm.
Packets 11-20:  Sleep starts declining. HRV drops slightly.
Packets 21-35:  Stress score rising. Recovery score falling.
                Sleep efficiency dropping. Steps decreasing.
Packets 36-40:  Agent has been asking questions.
                User reported "work has been intense."
                user_reported fields populated.
Packets 41-50:  Pattern fully confirmed. Agent delivers
                longitudinal insight about burnout trajectory.
```

### Scenario B — Emergency Event Arc (for elderly user)
```
Packets 1-15:   Calm baseline established. Normal evening.
                HR ~68, sedentary, 9pm, at home.
Packet 16:      HR jumps to 134. Activity state: sedentary.
                Time: 2:17am. SpO2: 93%. Breathing rate: 22.
                EDA: high.
Packet 17-18:   Pattern confirmed — not a spike, sustained.
                Agent fires Level 5 emergency protocol.
                Family notified via console.
```

---

# 6. Memory Architecture

## Overview

The memory system is the backbone of everything. All personalization, all longitudinal insights, all pattern recognition — it depends entirely on memory working well.

Two-layer architecture:

```
┌─────────────────────────────────────────────┐
│         LAYER 1: LIVING PROFILE              │
│         (LangGraph State — always in context)│
│                                              │
│  ~400 tokens. Compact. Always present in     │
│  every LLM call. Updated every 7 days        │
│  through summarization.                      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│         LAYER 2: EPISODIC MEMORY             │
│         (ChromaDB — retrieved via RAG)       │
│                                              │
│  Full archive. All significant health events,│
│  anomalies, conversations, weekly summaries. │
│  Only relevant pieces retrieved per query.  │
└─────────────────────────────────────────────┘
```

## Layer 1 — The Living Profile

### Full Structure

```
LIVING PROFILE — [User Name]
Last Updated: [timestamp]

--- IDENTITY ---
Name: [name]
Age: [age]
Known conditions: [list or "none reported"]
Current medications: [list or "none"]
Goals: [selected goals from onboarding]
Emergency contact: [name + contact method]

--- BASELINES ---
Status: [LEARNING (Days 1-14) | ESTABLISHED (Day 14+)]
Resting HR: [X] bpm
Typical HRV: [X] ms
Typical SpO2: [X]%
Typical skin temperature: [X]°C
Typical sleep duration: [X] hours
Typical sleep efficiency: [X]%
Typical daily steps: [X]
Typical resting breathing rate: [X]
Typical stress score (resting): [X]
Typical recovery score: [X]

--- CURRENT STATE ---
Sleep trend (7 day): [improving / stable / declining]
HRV trend (7 day): [improving / stable / declining]
Stress trend (7 day): [low / normal / elevated / high]
Recovery trend (7 day): [improving / stable / deteriorating]
Overall trajectory: [positive / stable / concerning / alarming]

--- KNOWN PATTERNS ---
[Bullet list of confirmed patterns discovered over time]
Examples:
- Elevated stress score consistently observed Monday-Wednesday afternoons
- Sleep onset takes longer than 40 mins when stress score above 65
- HRV drops significantly after days with under 5000 steps
- Recovery score below 50 correlates with next-day elevated resting HR

--- COMMUNICATION PROFILE ---
Preferred style: [brief/conversational/data-focused/gentle]
Directness: [clinical ←——→ conversational] score 1-5
Depth: [data-heavy ←——→ plain language] score 1-5
Tone: [formal ←——→ warm] score 1-5
Length preference: [detailed ←——→ brief] score 1-5
Alert sensitivity: [low / medium / high]
Best engagement times: [observed from response patterns]
Engagement patterns: [notes on what user responds well to]

--- CURRENT CONCERNS ---
[Active flags that haven't been resolved]
Examples:
- 4 consecutive nights under 5.5 hours sleep (Day 3 of pattern)
- HRV 34% below personal baseline this week
- Recovery score below 40 for 6 days straight
```

### How Baselines Are Established

**Days 1-14 — Learning Period:**
- Population averages for user's age and activity level used as temporary baselines
- Every incoming packet contributes to rolling baseline calculation
- Agent is transparent about this: *"I'm still learning your personal baselines — insights will sharpen over the next two weeks"*
- Only extreme deviations trigger alerts during this period

**Day 14+ — Personal Baselines Active:**
- Population averages discarded
- Personal baselines computed from 14 days of observed data
- All comparisons from this point are against the user's own history
- System switches automatically — no user action needed

**Baseline Computation Method:**
- Resting HR: median of all readings where `activity_state = "sedentary"` and `time_of_day = "morning"`
- HRV: 7-day rolling average of morning readings
- Sleep metrics: 14-day rolling averages
- Stress/Recovery scores: 7-day rolling averages
- Steps: 14-day rolling average

### Living Profile Update Rules

The Living Profile is updated in two ways:

**Continuous micro-updates (after every significant event):**
- Current concerns list updated
- Current state trends updated
- Communication profile adjusted from user responses

**Full 7-day summarization pass:**
- Pull all episodic memories from last 7 days from ChromaDB
- Recompute all baselines with new data included
- Identify new patterns that have emerged
- Update known patterns list
- Compress old raw episodes in ChromaDB into weekly summaries
- Preserve all individually significant episodes regardless of age

---

## Layer 2 — Episodic Memory (ChromaDB)

### Episodic Memory Document Structure

Every significant event stored in ChromaDB follows this structure:

```json
{
  "id": "episode_[timestamp]_[user_id]",
  "timestamp": "2025-04-25T23:14:00",
  "user_id": "user_a",
  "event_type": "anomaly | pattern | conversation | weekly_summary | onboarding",
  "metrics_snapshot": {
    "heart_rate": 92,
    "hrv": 28,
    "stress_score": 72,
    "recovery_score": 34
  },
  "context_snapshot": {
    "time_of_day": "night",
    "activity_state": "sedentary",
    "location_zone": "home"
  },
  "deviation_from_baseline": {
    "heart_rate": "+24% above personal baseline",
    "hrv": "-31% below personal baseline"
  },
  "significance": "pattern_confirmed | single_occurrence | resolved | ongoing",
  "agent_action_taken": "flagged_analyst | logged_only | asked_user | alerted_user | notified_family",
  "user_response": null,
  "outcome": null,
  "tags": ["sleep", "stress", "hr_elevated", "sedentary", "night"]
}
```

### RAG Query Strategy

The Analyst queries ChromaDB with two types of queries:

**Tag-based filtering:**
```python
# Find past episodes with similar signal combination
results = episodic_collection.query(
    query_texts=["elevated heart rate sedentary night"],
    where={"tags": {"$contains": "hr_elevated"}},
    n_results=3
)
```

**Semantic similarity search:**
```python
# Find semantically similar past situations
results = episodic_collection.query(
    query_texts=[current_situation_description],
    n_results=3
)
```

Always retrieve 2-3 most relevant past episodes — not more. Context window is precious.

### Memory Compression Schedule

```
Raw daily packets:        Kept for 7 days
Weekly summaries:         Kept for 4 weeks  
Monthly summaries:        Kept indefinitely
Individual significant    Kept indefinitely
episodes:                 (regardless of age)
```

A "significant episode" is any event that was Level 3 or above, any user conversation with meaningful content, any confirmed pattern detection.

---

# 7. Onboarding Flow

## Design Principles

- Conversational, not a form
- One question at a time
- Ask only what the agent cannot figure out from telemetry over time
- Takes approximately 2 minutes
- Sets up the cold-start Living Profile

## The Conversation

```
AGENT: "Hello! I'm your personal health agent. Before we get 
started, I'd like to learn a little about you so I can give 
you insights that are actually relevant to you specifically — 
not generic advice.

This will take about 2 minutes. Let's begin.

What's your name?"

USER: [responds]

AGENT: "How old are you, [name]?"

USER: [responds]

AGENT: "How would you describe your general activity level?

  1. Mostly sedentary (desk job, light movement through the day)
  2. Lightly active (some walking, occasional exercise)
  3. Moderately active (regular exercise 3-4 times a week)
  4. Very active (daily intense training or physical work)"

USER: [selects]

AGENT: "Do you have any known health conditions I should be 
aware of? For example — hypertension, diabetes, heart 
conditions, anxiety, sleep disorders, or anything else.

Type 'none' if nothing applies."

USER: [responds]

AGENT: "Are you currently on any medication that affects heart 
rate, sleep, or stress? For example — beta blockers, SSRIs, 
or blood pressure medication.

Type 'none' if not applicable."

[NOTE: This is critical. Medications change baselines significantly.
A user on beta blockers will have artificially suppressed HR.
Without this information, baselines will be computed incorrectly.]

USER: [responds]

AGENT: "What's the main thing you'd like me to help you with?

  1. Sleep quality
  2. Stress and mental load
  3. Fitness and recovery
  4. General health awareness
  5. All of the above
  6. None — just monitoring for now"

[NOTE: Option 6 is important. Some users want awareness without 
advice. If selected, the agent dials back proactive suggestions 
and only responds when directly asked or in emergencies.]

USER: [selects]

AGENT: "Last thing — how would you like me to communicate 
with you?

  1. Brief and direct. Just the important stuff.
  2. Conversational. Talk to me like a person.
  3. Data-focused. I like numbers and specifics.
  4. Gentle. I don't want to be alarmed unnecessarily."

USER: [selects]

AGENT: "One important question — would you like to add an 
emergency contact? This is someone I would notify if I ever 
detect something that needs immediate attention. This is 
especially important if you live alone or have any known 
heart or health conditions.

  1. Yes, add a contact
  2. Skip for now"

[If yes:]

AGENT: "What is their name? And how should I reach them?
(For now I'll log notifications to console.)"

USER: [responds]

AGENT: "Perfect. I'm all set and starting to monitor now.

For the first two weeks I'll be learning your personal 
baselines — what's normal specifically for you. My insights 
will get sharper as I get to know you better.

You can talk to me anytime by typing here. I'll also check 
in with you occasionally when I notice something worth 
discussing.

Let's get started, [name]."
```

## Post-Onboarding: Initial Living Profile

Immediately after onboarding, construct the first Living Profile:

```
LIVING PROFILE — [Name]
Generated: Day 1 — Onboarding

--- IDENTITY ---
Name: [from onboarding]
Age: [from onboarding]
Known conditions: [from onboarding]
Current medications: [from onboarding]
Goals: [from onboarding]
Emergency contact: [from onboarding or "not set"]

--- BASELINES ---
Status: LEARNING — Personal baselines not yet established
Using population estimates for age [X], activity level [Y]:
  Estimated resting HR: [range]
  Estimated HRV: [range]
  Estimated SpO2: ~97-99%
  Recommended sleep: 7-9 hours

--- CURRENT STATE ---
Insufficient data — monitoring started today.

--- KNOWN PATTERNS ---
None yet. Will emerge over time.

--- COMMUNICATION PROFILE ---
Style preference: [from onboarding]
All other dimensions: Default (medium sensitivity)
Engagement patterns: Not yet observed.

--- CURRENT CONCERNS ---
None.
```

---

# 8. Agent 1 — The Profiler

## Role

Build and continuously maintain the most accurate, up-to-date model of this specific person. Everything else in the system depends on this being good.

The Profiler is the only agent that receives the raw telemetry stream directly. It is always running.

## What The Profiler Does On Every Packet

```
1. Receive JSON packet
2. Parse all fields
3. Compare each vital to personal baseline in Living Profile
4. Compute deviation percentages
5. Update rolling state (current trends)
6. Check pattern buffer (see below)
7. Log to ChromaDB if significant
8. Activate Analyst if pattern confirmed
9. Update Living Profile current state section
```

## Significance Thresholds

The Profiler uses these thresholds to decide if a deviation is worth logging. All thresholds are relative to **personal baseline**, not population averages.

| Metric | Threshold |
|---|---|
| Heart Rate (resting) | More than 20% above personal resting baseline while sedentary |
| HRV | More than 25% below personal baseline |
| SpO2 | Drops below 94% |
| Sleep efficiency | Drops below 60% |
| Stress score | Above 75 while sedentary |
| Recovery score | Below 30 |
| Skin temperature | More than 0.8°C above personal baseline |
| Breathing rate | More than 25% above personal baseline while sedentary |

During the learning period (Days 1-14), thresholds are more conservative — only extreme deviations (40%+) trigger logging.

## The Pattern Buffer

A single anomalous reading is noise. A confirmed pattern is signal.

The Profiler maintains a short-term pattern buffer:

```python
pattern_buffer = {
    "hr_elevated": 0,      # count of consecutive packets with elevated HR
    "hrv_low": 0,
    "stress_elevated": 0,
    "spo2_low": 0,
    # etc.
}
```

Rules:
- Buffer count increments on each packet where threshold is exceeded
- Buffer count resets to 0 when reading returns to normal
- **Pattern confirmed at count = 3** (3 consecutive anomalous packets)
- Confirmed pattern triggers Analyst activation
- Single spikes are logged to ChromaDB with tag `single_occurrence` but do not trigger Analyst

## ChromaDB Logging

Every significant event gets logged. Format:

```json
{
  "id": "episode_20250425231400_user_a",
  "timestamp": "2025-04-25T23:14:00",
  "user_id": "user_a",
  "event_type": "anomaly",
  "metrics_snapshot": {
    "heart_rate": 92,
    "hrv": 28,
    "stress_score": 72,
    "recovery_score": 34,
    "activity_state": "sedentary"
  },
  "context_snapshot": {
    "time_of_day": "night",
    "location_zone": "home",
    "day_of_week": "Thursday"
  },
  "deviation_from_baseline": {
    "heart_rate": "+24.3%",
    "hrv": "-31.2%"
  },
  "significance": "pattern_confirmed",
  "agent_action_taken": "flagged_analyst",
  "user_response": null,
  "outcome": null,
  "tags": ["hr_elevated", "hrv_low", "sedentary", "night", "stress"]
}
```

## The 7-Day Summarization Pass

Triggered automatically every 7 days. Steps:

```
1. Pull all episodic memories from last 7 days from ChromaDB
2. Send to LLM with instruction to:
   a. Identify patterns that appeared this week
   b. Compute updated averages for all metrics
   c. Note any resolved concerns
   d. Note any new concerns
   e. Identify changes from previous week
3. Update Living Profile with findings
4. Compress raw daily episodes into weekly summary document in ChromaDB
5. Preserve individual episodes tagged as Level 3+ significance
6. Log new weekly summary to ChromaDB
```

## Profiler Initiated Questions

The Profiler can initiate a question to the user when it encounters genuine uncertainty — situations where telemetry data alone cannot explain what it is seeing.

Trigger conditions:
- Anomaly detected but no historical context to compare to
- Pattern detected that contradicts previous known patterns
- Metrics suggest possible external life factor (work stress, illness, travel)
- User has not responded to a previous question about an ongoing concern

Question rules:
- One question at a time — never two
- Specific, not vague
- Always gives user an easy out
- Response stored in ChromaDB and used to update Living Profile

Example questions:
```
"Your sleep has been under 5 hours for 4 nights. 
Is this unusual for you, or has something changed recently?"

"I've noticed your heart rate tends to spike around 3pm on 
weekdays. Is that typically a stressful time for you?"

"Your recovery has been poor all week. How are you feeling 
in general — any changes in diet or life stress lately?"
```

---

# 9. Agent 2 — The Analyst

## Role

Look at everything the Profiler knows — current state, personal history, patterns, clinical context — and decide: **is there something worth doing right now?** And if yes, **what exactly?**

## Activation Conditions

The Analyst activates when:
1. Profiler flags a confirmed pattern (3+ consecutive anomalous packets)
2. User sends a message (user-initiated conversation)
3. Scheduled check-in time (e.g., morning summary if configured)

## The Decision Framework — Step By Step

### Step 1 — Understand What Happened
Read the current packet, the Profiler's flag, and the current state section of the Living Profile. What is the triggering signal? What is the context right now?

### Step 2 — Is This Real or Noise?
Was this a confirmed pattern (3+ packets) or a single spike?
- Single spike → log only, do not proceed
- Confirmed pattern → continue to Step 3

### Step 3 — Query Personal History (RAG)
Query ChromaDB episodic memory:
- Has this exact pattern happened before for this user?
- What was the context at that time?
- What happened in the days following that episode?
- Did the agent act? What was the outcome?
- Did the user respond? What did they say?

Retrieve 2-3 most relevant past episodes. These go into the LLM context.

### Step 4 — Query Medical Knowledge Base (RAG)
Query ChromaDB medical knowledge base:
- What does this combination of signals suggest clinically?
- Is this potentially serious or likely benign?
- Are there known health implications for this pattern?
- What is the evidence-based recommended response?

Retrieve 1-2 most relevant knowledge base entries.

### Step 5 — Severity Classification

Classify the current situation into one of five levels:

```
LEVEL 1 — INSIGHT
Definition: Interesting pattern worth sharing. Not urgent. Not actionable immediately.
Example: "Your HRV tends to drop on days you skip exercise. 
         This has happened 5 times in the past 3 weeks."
Timing: Queue for next natural conversation moment. 
        Do NOT interrupt user for this.
Agent action: Communicate at appropriate time (morning, 
              or when user initiates conversation)

LEVEL 2 — NUDGE
Definition: Something actionable the user should know now but not alarming.
Example: "Your recovery score has been low all week. 
         Consider lighter activity today."
Timing: Send within the hour. Respect 2-hour guideline 
        (see Timing Rules below).
Agent action: Gentle notification. Conversational tone.

LEVEL 3 — CONCERN
Definition: Something that warrants attention and possibly needs 
            context from the user before concluding.
Example: "HR elevated + sedentary + 1am + poor sleep streak. 
         Need more context to understand this."
Timing: Send now. Override 2-hour guideline if needed 
        (acknowledge it if overriding).
Agent action: Ask user a specific question. 
              Wait for response before escalating.

LEVEL 4 — ALERT
Definition: Something abnormal enough to flag clearly. 
            Not immediately life-threatening but needs 
            user awareness now.
Example: "Your stress indicators have been in the red 
         for 8 consecutive days. This needs addressing."
Timing: Send immediately. No timing restrictions apply.
Agent action: Direct notification. Specific suggested action. 
              Recommend consulting doctor if pattern persists.

LEVEL 5 — EMERGENCY
Definition: Vitals suggest potential medical event. 
            Especially critical in vulnerable users.
Example: "HR 142, completely sedentary, 2am, SpO2 dropping, 
         elderly user, no movement for 45 minutes."
Timing: Send immediately. No restrictions.
Agent action: Immediate user alert + family/emergency 
              contact notification via console.
```

### Step 6 — Timing and Context Check

Even if something is worth saying — is right now the right time?

```
IS USER SLEEPING?
  → Level 1, 2, 3: Hold. Do not interrupt sleep.
  → Level 4: Send anyway.
  → Level 5: Send immediately.

IS BATTERY BELOW 10% AND USER IN UNKNOWN LOCATION?
  → Level 1, 2: Hold. Don't add stress.
  → Level 3+: Send anyway.

HAS AGENT SENT A MESSAGE IN LAST 2 HOURS?
  → Level 1: Hold until next natural moment.
  → Level 2: Hold unless situation is new and different.
  → Level 3: Send anyway. Acknowledge the recency:
             "I know I mentioned something earlier, but..."
  → Level 4, 5: Send immediately. No question.

IS THIS A LEVEL 1 INSIGHT AND IT IS 3AM?
  → Queue for morning delivery.
```

### Step 7 — Draft Content Substance

The Analyst drafts the *substance* of what needs to be communicated:
- The key facts
- The relevant historical comparison (from RAG)
- The clinical context (from RAG)
- The recommended action
- Whether a question needs to be asked
- Whether family notification is needed

This substance is passed to the Communicator — **not** the final wording. The Communicator handles wording.

## What The Analyst Passes To The Communicator

```json
{
  "severity_level": 3,
  "trigger": "hr_elevated_pattern + hrv_low + sedentary + night",
  "key_facts": [
    "HR 34% above personal baseline",
    "HRV 28% below personal baseline", 
    "Pattern confirmed across 4 consecutive packets",
    "Current time: 1:14am",
    "Activity: sedentary"
  ],
  "historical_context": "Similar pattern occurred Oct 12 and Nov 3. Both preceded 2-3 days of reported fatigue.",
  "clinical_context": "Elevated resting HR combined with low HRV at rest during night hours can indicate accumulated stress load or early illness onset.",
  "recommended_action": "Ask user about current wellbeing. Monitor for next 30 minutes.",
  "question_to_ask": "Are you awake right now? How are you feeling?",
  "notify_family": false,
  "add_to_current_concerns": true
}
```

---

# 10. Agent 3 — The Communicator

## Role

Take what the Analyst concluded and deliver it in exactly the right way for this specific person at this specific moment.

The Communicator knows the *what* from the Analyst. It decides the *how.*

## Communication Style Dimensions

Each user has a profile on five dimensions, scored 1-5:

```
Directness:   1=Clinical/formal  ←————→  5=Conversational/casual
Depth:        1=Data-heavy       ←————→  5=Plain language only
Tone:         1=Professional     ←————→  5=Warm/friendly
Length:       1=Detailed         ←————→  5=Very brief
Framing:      1=Warning-forward  ←————→  5=Insight/curiosity-forward
```

Initial values set from onboarding answer. Updated continuously from engagement signals.

## How Communication Profile Is Updated

The Communicator does NOT ask "how do you like to be spoken to?" It learns from behavior:

| Signal | Update |
|---|---|
| User engages with data-heavy response | Nudge Depth toward 1 (data-heavy) |
| User gives short dismissive response to clinical message | Nudge Directness toward 5 (conversational) |
| User responds positively to questions | Ask more questions going forward |
| User ignores or dismisses long messages | Nudge Length toward 5 (brief) |
| User responds better in evening than morning | Note in engagement patterns |
| User responds with anxiety to health data | Nudge Framing toward 5 (insight-forward, not warning-forward) |
| User explicitly requests something | Override profile, honor the request |

Updates are small increments — not sudden swings. The profile shifts gradually.

## Same Insight, Three Communication Styles

To illustrate how the Communicator works, here is the same Analyst output delivered three different ways:

**Analyst substance:** *HRV declined 30% over 7 days. Sleep efficiency also declining. Stress score consistently elevated. Pattern suggests accumulating stress load.*

**Style 1 — Direct, data-aware, brief (scores: 1,2,1,4,2):**
> "Your HRV is down 30% from your baseline this week and your sleep efficiency has dropped with it. Your body is accumulating stress faster than it's recovering. Worth taking seriously."

**Style 2 — Warm, plain language, conversational (scores: 5,5,5,3,5):**
> "Hey, I've been noticing something over the past week — your body has been showing signs of building up stress without enough recovery time. Your sleep hasn't been as restorative lately either. How are you feeling in general?"

**Style 3 — Gentle, simple, caring (scores: 4,5,5,4,5):**
> "I want to check in with you. For the past week your body has been showing signs of tiredness and stress. Are you getting enough rest? Is everything okay?"

## Proactive Question Formulation Rules

When Analyst requests a question be asked:

1. **One question only** — never two in the same message
2. **Specific not vague** — "Is work particularly stressful this week?" not "How are you?"
3. **Give easy out** — "Is this normal for you, or does something feel off?"
4. **Acknowledge the interruption if Level 3 override** — "I know I checked in earlier, but I want to ask..."
5. **Store response** — user answer goes to ChromaDB and updates Living Profile
6. **Follow up** — if user answer reveals new information, Profiler updates the profile

## Disclaimer Logic

For any health-related insight at Level 3 or above, the Communicator always appends (adapted to style):

> "If this pattern continues, it's worth mentioning to your doctor. I can help you track it in the meantime."

This is non-negotiable. The agent is a pattern recognizer and advisor — not a clinician. It does not diagnose. It does not prescribe.

## Family Notification Format (Console)

When Level 5 is triggered:

```
========================================
⚠️  EMERGENCY ALERT
========================================
Time: 2025-04-25 02:17:43
User: [Name]
Contact: [Emergency contact name]

SITUATION:
Heart rate: 134 bpm (97% above personal baseline)
Activity: Completely sedentary
SpO2: 93% (below safe threshold)
Time: 2:17am
Pattern duration: 4 consecutive readings (approx 8 minutes)

AGENT ASSESSMENT:
Abnormal cardiac indicators detected during sleep hours.
This combination is not consistent with normal sleep or 
resting state for this user.

ACTION RECOMMENDED:
Immediate check-in with user recommended.
Consider contacting emergency services if user 
is unresponsive.

[This is an automated alert from the Health Agent]
========================================
```

---

# 11. Interaction Modes

## Mode 1 — Agent Observes, Stays Silent
Profiler sees data. Nothing crosses significance threshold. Logs to ChromaDB. No action. This is the most common mode — the agent is always watching, usually quietly.

## Mode 2 — Agent Initiated (Levels 1-5)
Profiler detects pattern. Analyst classifies and decides. Communicator delivers. See severity levels above for full behavior per level.

## Mode 3 — Agent Asks A Question
Profiler encounters uncertainty. Cannot explain what it is seeing from telemetry alone. Formulates a specific question. Waits for response. Updates profile from answer.

## Mode 4 — User Initiated Conversation
User talks to the agent at any time. Agent always responds — regardless of 2-hour guideline. The 2-hour guideline applies only to **agent-initiated** interruptions.

### Types Of User Questions The Agent Handles

**Current state questions:**
- "How am I doing today?"
- "What do my vitals look like right now?"
- "Am I stressed?"

Response approach: Analyst pulls current Living Profile state + most recent telemetry packet. Communicator delivers personalized response comparing current state to personal baseline.

**Explanatory questions:**
- "Why do I feel tired?"
- "What does low HRV mean for me?"
- "Is this normal for me?"

Response approach: RAG on medical knowledge base for clinical context + RAG on personal history for individual comparison. Answer is both clinically grounded and personally relevant.

**Historical questions:**
- "How has my sleep been this week?"
- "Have I been more stressed than usual lately?"
- "When was the last time my recovery was good?"

Response approach: ChromaDB query for relevant historical episodes. Compute trend from stored data. Deliver with personal context.

**Advice questions:**
- "Should I work out today?"
- "What should I do about my sleep?"
- "Is there anything I should be worried about?"

Response approach: Analyst runs full decision framework on current state. Delivers recommendation grounded in personal history and clinical knowledge base.

**Conversational check-ins:**
- "I've been feeling off today"
- "I had a really stressful day"
- "I think I'm getting sick"

Response approach: Acknowledge what the user said. Cross-reference against telemetry data. Ask one clarifying question if helpful. Update Living Profile with the new context. Store in ChromaDB.

### Every User-Initiated Response Is Logged

All user conversations get stored as episodic memories in ChromaDB. This means the agent builds context not just from vitals — but from what the user says about themselves over time.

---

# 12. Medical Knowledge Base

## Purpose

Gives the Analyst clinical grounding. The difference between the agent saying something evidence-based versus just pattern matching on numbers. Prevents hallucination by providing retrieved facts rather than generated clinical claims.

## Important Disclaimer Built Into Every Clinical Insight

The agent does not diagnose. It does not prescribe. Every clinical insight includes a recommendation to consult a doctor if the pattern persists. This is built into the Communicator's output logic — not optional.

## Structure Of Each Knowledge Base Entry

```json
{
  "id": "kb_001",
  "domain": "heart_rate_hrv | sleep | stress_recovery | spo2_breathing | temperature | combined_patterns",
  "signals_involved": ["hr_elevated", "hrv_low", "sedentary"],
  "duration_context": "single_occurrence | multi_day | sustained",
  "user_context": "any | elderly | athlete | sedentary | medicated",
  "title": "Short descriptive title",
  "interpretation": "What this pattern means clinically",
  "severity_suggestion": 1,
  "recommended_agent_action": "What the agent should do",
  "what_not_to_conclude": "Common misinterpretations to avoid",
  "sources": ["peer_reviewed", "clinical_guidelines", "consumer_health"]
}
```

## Domain 1 — Heart Rate & HRV

Key entries to include:

- Elevated resting HR — stress vs dehydration vs illness onset vs overtraining vs anxiety (how to differentiate by context)
- Low HRV — sympathetic nervous system dominance, poor recovery, chronic stress
- HRV single-day drop vs multi-day decline (very different implications)
- Exercise-induced HR elevation vs stress-induced (activity_state differentiates)
- Sustained elevated resting HR while sedentary — when to escalate
- Age-adjusted HR context (what is normal at 25 vs 65)
- Bradycardia context (very low HR — when it is normal for athletes vs concerning)

## Domain 2 — Sleep

Key entries to include:

- Sleep debt accumulation and its cognitive and physical effects
- What poor sleep efficiency indicates (time in bed vs actual sleep)
- Sleep onset latency above 30 minutes — stress and anxiety relationship
- Low deep sleep percentage — physical recovery implications
- Low REM percentage — emotional regulation and memory implications
- Chronic sleep deprivation markers (cumulative nights under threshold)
- How sleep quality predicts next-day HRV and recovery score
- Sleep apnea indicators from breathing rate and wakefulness patterns

## Domain 3 — Stress & Recovery

Key entries to include:

- HRV as stress and recovery marker — the physiological mechanism
- Sustained high stress scores — what they indicate over time
- Burnout progression: early stage markers, mid-stage, late stage
- The relationship between recovery score and safe exertion level
- Overtraining syndrome — markers and risks
- Acute stress vs chronic stress differentiation from vitals
- The interaction between stress, sleep, and HRV (the triangle)

## Domain 4 — SpO2 & Breathing

Key entries to include:

- Normal SpO2 ranges by context
- Below 95% — when to note, when to act
- Below 90% — emergency threshold
- Elevated breathing rate at rest — stress vs illness vs respiratory issue
- Sleep apnea indicators
- Weather/altitude effects on SpO2 (contextual awareness)

## Domain 5 — Temperature

Key entries to include:

- Elevated skin temperature as early illness indicator (often precedes symptoms by 1-2 days)
- Normal intraday temperature variability
- Temperature + elevated HR + reduced HRV — illness onset signature
- Fever threshold indicators from wearable skin temperature

## Domain 6 — Combined Signal Patterns (Most Important)

This is what makes the Analyst powerful. Not single metrics — combinations. Every entry here describes a multi-signal pattern with specific context.

### Key Combined Patterns To Include:

**Burnout Trajectory:**
```
Signals: HRV declining trend (7+ days) + Resting HR rising trend + 
         Sleep efficiency declining + Stress score consistently elevated
Duration: Multi-day (7+)
Interpretation: Classic burnout progression. Body under sustained 
                load without adequate recovery. 
Severity suggestion: 4
Action: Direct conversation. Acknowledge pattern clearly. 
        Recommend life audit and doctor consultation.
```

**Stress-Driven Insomnia:**
```
Signals: Poor sleep efficiency + High sleep onset latency + 
         High evening stress score + Normal daytime activity
Duration: Multi-day (3+)
Interpretation: Stress interfering with sleep onset. Anxiety-driven 
                insomnia pattern.
Severity suggestion: 3
Action: Ask about evening stress sources. Suggest wind-down strategies.
```

**Overtraining:**
```
Signals: Low recovery score + High active minutes + 
         Declining HRV trend
Duration: Multi-day
Interpretation: User exercising despite inadequate recovery. 
                Overtraining risk.
Severity suggestion: 2
Action: Gentle nudge toward rest day. Explain recovery science briefly.
```

**Illness Onset Signature:**
```
Signals: Elevated skin temperature + Elevated resting HR + 
         Reduced HRV + Elevated breathing rate
Duration: Single occurrence to multi-day
Interpretation: Early illness markers. Often precedes symptoms 
                by 24-48 hours.
Severity suggestion: 3
Action: Flag pattern. Ask how user is feeling. 
        Suggest rest and monitoring.
```

**Cardiac Concern — Elderly User:**
```
Signals: HR significantly elevated + Completely sedentary + 
         Night time hours + SpO2 declining + User age 60+
Duration: Sustained (3+ packets)
Interpretation: Abnormal cardiac indicators during sleep/rest hours 
                for elderly user. Not explainable by activity or 
                normal stress.
Severity suggestion: 5
Action: Immediate user alert + family notification. 
        Recommend emergency services if unresponsive.
```

**Sleep Apnea Indicators:**
```
Signals: SpO2 dipping + Elevated breathing rate + 
         Multiple wake-ups + Poor sleep efficiency
Duration: Recurring across multiple nights
Interpretation: Pattern consistent with sleep-disordered breathing. 
                Possible sleep apnea.
Severity suggestion: 4
Action: Flag clearly. Recommend sleep study consultation.
```

**Accumulated Stress Load:**
```
Signals: Elevated HR + Low HRV + Poor sleep (3+ days combined)
Duration: Multi-day
Interpretation: Body not recovering between stress cycles. 
                Accumulated load building. Risk of burnout if continued.
Severity suggestion: 3
Action: Flag trend. Ask about life stressors. 
        Suggest recovery-focused actions.
```

## Knowledge Base Size For Hackathon

- 6 domains
- 10-15 entries per single-signal domain
- 20-30 combined pattern entries
- Total: approximately 100-120 documents

These are written as structured text/JSON files, embedded into ChromaDB at application startup using sentence-transformers embeddings. The RAG retrieval handles the rest.

---

# 13. Complete System Flow

## End-To-End Flow Diagram

```
APPLICATION STARTUP
  ↓
Load medical knowledge base into ChromaDB
Load existing Living Profile (or create new)
Initialize LangGraph state with Living Profile
Start telemetry stream listener
  ↓

─────────────────────────────────────────
NEW USER PATH (no existing profile)
─────────────────────────────────────────
  ↓
Run onboarding conversation
Build initial Living Profile from responses
Store onboarding episode in ChromaDB
Begin monitoring
  ↓

─────────────────────────────────────────
MAIN MONITORING LOOP (continuous)
─────────────────────────────────────────
  ↓
JSON packet arrives from telemetry stream
  ↓
PROFILER ACTIVATES
  ↓
  ├─ Compare vitals to Living Profile baselines
  ├─ Compute deviations
  ├─ Update rolling state in Living Profile
  ├─ Update pattern buffer
  │
  ├─ [DEVIATION BELOW THRESHOLD]
  │     └─ Log packet summary. Continue loop.
  │
  ├─ [DEVIATION ABOVE THRESHOLD, BUFFER < 3]
  │     └─ Log to ChromaDB (single_occurrence tag)
  │        Continue monitoring.
  │
  └─ [PATTERN CONFIRMED — buffer reaches 3]
        ↓
        Log to ChromaDB (pattern_confirmed tag)
        Activate Analyst
          ↓

─────────────────────────────────────────
ANALYST DECISION FLOW
─────────────────────────────────────────
  ↓
Read current Living Profile (always in context)
  ↓
RAG Query 1: ChromaDB episodic memory
  "Has this happened before for this user?"
  Retrieve top 2-3 relevant past episodes
  ↓
RAG Query 2: ChromaDB medical knowledge base
  "What does this combination mean clinically?"
  Retrieve top 1-2 relevant entries
  ↓
Run Decision Framework (Steps 1-7)
  ↓
Classify severity: Level 1 / 2 / 3 / 4 / 5
  ↓
Run timing and context check
  ↓
  ├─ [NOT RIGHT TIME]
  │     └─ Queue message for appropriate time
  │        Return to monitoring loop
  │
  └─ [PROCEED]
        ↓
        Draft content substance
        Determine if question needed
        Determine if family notification needed
        Pass to Communicator
          ↓

─────────────────────────────────────────
COMMUNICATOR OUTPUT FLOW
─────────────────────────────────────────
  ↓
Read communication profile from Living Profile
  ↓
Adapt tone, style, length, framing
  ↓
Formulate final message
  ↓
  ├─ [LEVEL 1-4] Deliver message to user interface
  │
  └─ [LEVEL 5] Deliver emergency message to user
               + Trigger family notification console output
  ↓
Log interaction to ChromaDB
  ↓

─────────────────────────────────────────
USER RESPONSE (if any)
─────────────────────────────────────────
  ↓
User responds to agent message or question
  ↓
Response stored in ChromaDB (updates episode record)
  ↓
Profiler receives response
  ↓
  ├─ New information revealed?
  │     └─ Update Living Profile accordingly
  │
  └─ Confirms existing understanding?
        └─ Update confidence in current pattern
  ↓
Return to monitoring loop

─────────────────────────────────────────
USER INITIATED CONVERSATION (anytime)
─────────────────────────────────────────
  ↓
User types message
  ↓
Goes directly to Analyst (bypasses Profiler pattern detection)
  ↓
Analyst reads Living Profile + queries ChromaDB for relevant history
Analyst queries medical KB if question is health-explanatory
  ↓
Analyst drafts response substance
  ↓
Communicator delivers in user's preferred style
  ↓
Conversation logged to ChromaDB
Living Profile updated if new information revealed
  ↓
Return to monitoring loop

─────────────────────────────────────────
7-DAY SUMMARIZATION PASS (scheduled)
─────────────────────────────────────────
  ↓
Pull all episodic memories from last 7 days
  ↓
LLM summarization pass:
  - Identify patterns this week
  - Compute updated baselines
  - Note resolved and new concerns
  - Compare to previous week
  ↓
Update Living Profile (full refresh)
  ↓
Compress raw episodes into weekly summary in ChromaDB
Preserve all Level 3+ episodes individually
  ↓
System is now sharper than it was 7 days ago
  ↓
Return to monitoring loop
```

---

# 14. LangGraph Implementation Guide

## Graph Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

# Define the state that flows through the graph
class HealthAgentState(TypedDict):
    # Always present
    living_profile: dict
    current_packet: dict
    
    # Set by Profiler
    deviation_detected: bool
    pattern_confirmed: bool
    pattern_details: Optional[dict]
    
    # Set by Analyst
    severity_level: Optional[int]
    analyst_output: Optional[dict]
    proceed_to_communicate: bool
    
    # Set by Communicator
    final_message: Optional[str]
    notify_family: bool
    
    # Conversation
    user_message: Optional[str]
    agent_response: Optional[str]
```

## Node Definitions

```python
# Three main nodes
def profiler_node(state: HealthAgentState) -> HealthAgentState:
    # Process incoming telemetry packet
    # Compare to baselines in living_profile
    # Update pattern buffer
    # Set deviation_detected and pattern_confirmed
    # Log to ChromaDB if significant
    pass

def analyst_node(state: HealthAgentState) -> HealthAgentState:
    # RAG query ChromaDB episodic memory
    # RAG query medical knowledge base
    # Run decision framework
    # Set severity_level and analyst_output
    # Set proceed_to_communicate
    pass

def communicator_node(state: HealthAgentState) -> HealthAgentState:
    # Read communication profile from living_profile
    # Adapt message from analyst_output
    # Set final_message
    # Set notify_family if Level 5
    pass
```

## Conditional Routing

```python
def route_from_profiler(state: HealthAgentState) -> str:
    if state["pattern_confirmed"]:
        return "analyst"
    else:
        return END  # Just logged, no further action

def route_from_analyst(state: HealthAgentState) -> str:
    if state["proceed_to_communicate"]:
        return "communicator"
    else:
        return END  # Queued or not worth surfacing

# Build the graph
graph = StateGraph(HealthAgentState)
graph.add_node("profiler", profiler_node)
graph.add_node("analyst", analyst_node)
graph.add_node("communicator", communicator_node)

graph.set_entry_point("profiler")
graph.add_conditional_edges("profiler", route_from_profiler)
graph.add_conditional_edges("analyst", route_from_analyst)
graph.add_edge("communicator", END)

app = graph.compile()
```

## Groq API Integration Per Agent

```python
from groq import Groq

# Each agent gets its own client (separate API keys)
profiler_client = Groq(api_key=os.environ["GROQ_API_KEY_PROFILER"])
analyst_client = Groq(api_key=os.environ["GROQ_API_KEY_ANALYST"])
communicator_client = Groq(api_key=os.environ["GROQ_API_KEY_COMMUNICATOR"])

# Each agent call follows this pattern
def call_agent_llm(client, system_prompt, user_content, model="llama3-70b-8192"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,  # Lower temperature for health analysis = more consistent
        max_tokens=500
    )
    return response.choices[0].message.content
```

## ChromaDB Setup

```python
import chromadb
from sentence_transformers import SentenceTransformer

# Initialize
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Two collections
episodic_collection = chroma_client.get_or_create_collection(
    name="episodic_memory",
    metadata={"hnsw:space": "cosine"}
)

knowledge_collection = chroma_client.get_or_create_collection(
    name="medical_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# Query function
def query_memory(collection, query_text, n_results=3, where=None):
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where
    )
    return results
```

---

# 15. Folder Structure

```
adaptive_health_agent/
│
├── main.py                    # Application entry point
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env                       # API keys (never commit this)
│
├── agents/
│   ├── __init__.py
│   ├── profiler.py            # Profiler agent logic
│   ├── analyst.py             # Analyst agent logic
│   └── communicator.py        # Communicator agent logic
│
├── graph/
│   ├── __init__.py
│   ├── state.py               # LangGraph state definition
│   └── graph.py               # Graph construction and routing
│
├── memory/
│   ├── __init__.py
│   ├── living_profile.py      # Living Profile read/write/update
│   ├── episodic_memory.py     # ChromaDB episodic memory operations
│   └── summarizer.py          # 7-day summarization pass logic
│
├── knowledge_base/
│   ├── loader.py              # Load KB documents into ChromaDB
│   └── documents/
│       ├── heart_rate_hrv.json
│       ├── sleep.json
│       ├── stress_recovery.json
│       ├── spo2_breathing.json
│       ├── temperature.json
│       └── combined_patterns.json
│
├── telemetry/
│   ├── stream.py              # Mock JSON stream generator
│   ├── user_a_scenario.py     # Stress accumulation arc
│   └── user_b_scenario.py     # Emergency event arc
│
├── onboarding/
│   └── onboarding.py          # Onboarding conversation flow
│
└── utils/
    ├── baseline_calculator.py  # Compute personal baselines from history
    ├── pattern_buffer.py       # Pattern confirmation logic
    └── notifications.py        # Console notification formatter
```

---

# 16. Build Order

Build in this exact order. Each step is testable before moving to the next.

## Phase 1 — Foundation (Build This First)

**Step 1: Environment setup**
- Install all dependencies
- Set up .env with Groq API keys
- Verify Groq API connection with a simple test call

**Step 2: Mock telemetry stream**
- Build `telemetry/stream.py`
- It emits one JSON packet every 5 seconds
- Build User A scenario first (stress accumulation arc)
- Test: run the script, verify packets print to console correctly

**Step 3: ChromaDB setup**
- Initialize both collections
- Test: add one document, query it back, verify embedding works

**Step 4: Medical knowledge base**
- Write knowledge base documents (start with combined_patterns — most important)
- Build `knowledge_base/loader.py`
- Load documents into ChromaDB on startup
- Test: query "elevated heart rate sedentary night" — verify relevant documents return

## Phase 2 — Memory Layer

**Step 5: Living Profile**
- Build `memory/living_profile.py`
- Functions: create, read, update, get_baselines, update_baselines
- Test: create a profile, update a field, read it back

**Step 6: Episodic memory operations**
- Build `memory/episodic_memory.py`
- Functions: log_episode, query_similar, get_recent, update_episode
- Test: log a mock episode, query it back with semantic search

**Step 7: Onboarding flow**
- Build `onboarding/onboarding.py`
- Runs the conversation, collects answers, builds initial Living Profile
- Test: run onboarding, verify Living Profile is correctly populated

## Phase 3 — Agents

**Step 8: Profiler agent**
- Build `agents/profiler.py`
- Implement baseline comparison and deviation detection
- Implement pattern buffer
- Implement ChromaDB logging
- Test: feed it 5 normal packets, then 3 anomalous packets — verify pattern confirms on packet 3

**Step 9: Analyst agent**
- Build `agents/analyst.py`
- Implement decision framework (all 7 steps)
- Implement severity classification
- Implement RAG queries to both ChromaDB collections
- Test: give it a mock pattern, verify it returns correct severity level and substance

**Step 10: Communicator agent**
- Build `agents/communicator.py`
- Implement style adaptation from communication profile
- Implement message generation for all 5 severity levels
- Implement family notification for Level 5
- Test: give it Analyst output at each severity level, verify output style matches profile

## Phase 4 — Graph & Integration

**Step 11: LangGraph graph**
- Build `graph/state.py` and `graph/graph.py`
- Wire all three agents as nodes
- Implement conditional routing
- Test: run one full packet through the graph end-to-end

**Step 12: Full integration test**
- Connect telemetry stream to graph input
- Run User A scenario (stress arc) — verify agent responds correctly at the right moment
- Run User B scenario (emergency arc) — verify Level 5 triggers correctly with family notification

## Phase 5 — UI & Polish

**Step 13: Streamlit UI**
- Build `app.py`
- Show: live telemetry feed, agent messages, conversation input, Living Profile summary panel
- Test: run full demo flow through the UI

**Step 14: Summarization pass**
- Build `memory/summarizer.py`
- Pre-load User A with 7 days of history in ChromaDB
- Run summarization pass — verify Living Profile updates correctly
- This demonstrates the longitudinal memory system working

**Step 15: Final testing and cleanup**
- Run both full scenario arcs end to end
- Verify all ChromaDB queries return relevant results
- Verify communication style adapts correctly
- Clean up console output and UI presentation

---

# Appendix — Environment Variables

```
# .env file
GROQ_API_KEY_PROFILER=your_key_here
GROQ_API_KEY_ANALYST=your_key_here
GROQ_API_KEY_COMMUNICATOR=your_key_here

# ChromaDB (local, no key needed)
CHROMA_DB_PATH=./chroma_db

# Application settings
TELEMETRY_INTERVAL_SECONDS=5
PATTERN_BUFFER_THRESHOLD=3
BASELINE_LEARNING_DAYS=14
SUMMARIZATION_INTERVAL_DAYS=7
MESSAGE_COOLDOWN_MINUTES=120
```

---

# Appendix — Requirements.txt

```
langgraph
langchain
langchain-groq
groq
chromadb
sentence-transformers
streamlit
python-dotenv
```

---

*Document Version 1.0 — Complete planning and architecture for Adaptive Personal Health Agent*
*Built for Agentic AI Hackathon — IoT Wearable Context Engine Track*