# 🫀 AVA — Adaptive Virtual Assistant

**A wearable health advisor that learns how you like to be spoken to.**

AVA watches a live stream of smartwatch telemetry, compares every reading against *your* personal baseline, and only speaks up when something genuinely matters — in a tone tailored to you. A 3-agent LangGraph pipeline does the thinking; a Streamlit dashboard shows it happening in real time.

[![Live Demo](https://img.shields.io/badge/Live_Demo-ava--healthagent.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://ava-healthagent.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-orange.svg)
![Groq](https://img.shields.io/badge/Groq-GPT--OSS_20B-black.svg)

### 👉 [Try the live demo](https://ava-healthagent.streamlit.app/)

---

## 📖 Contents

- [Features](#-features)
- [How it works](#-how-it-works)
- [Using the app](#-using-the-app)
- [Run locally](#-run-locally)
- [Deploy to Streamlit Cloud](#-deploy-to-streamlit-cloud)
- [Configuration](#-configuration)
- [Project structure](#-project-structure)

## 🌟 Features

| | |
|---|---|
| 📡 **Live telemetry stream** | Simulated wearable feed — heart rate, HRV, SpO₂, breathing, stress, recovery, sleep — processed packet by packet. |
| 🎯 **Baseline-aware vitals** | Each reading is shown against the user's own baseline and flagged *Normal / Watch / Alert*, with trend charts and a dashed baseline line. |
| 🧬 **The Living Profile** | A persistent state object tracking physical baselines and a *communication profile* (directness, depth, tone, length, framing) that shifts based on how you reply. |
| 🧠 **Episodic memory (RAG)** | ChromaDB stores anomalies and conversations so the agent can recall past events ("you had a similar stress spike last week"). |
| 💸 **Cost-aware by design** | Cheap math detects deviations; a **pattern buffer** (3 consecutive anomalies) must confirm before any LLM is woken up. A cooldown stops message spam. |
| 🚨 **Escalation** | Severity levels 1–5. Level 5 produces an emergency message and notifies the user's emergency contact. |
| 📱 **Responsive UI** | Dark dashboard with chat and vitals side by side on desktop, stacked on mobile. |

## 🧠 How it works

```text
 telemetry packet
        │
        ▼
 ┌─────────────┐  deviation +   ┌─────────────┐  severity 1–5  ┌───────────────┐
 │  Profiler   │ ─ pattern ───▶ │   Analyst   │ ─ + context ─▶ │ Communicator  │ ─▶ message to user
 │ (watcher)   │   confirmed    │ (the doctor)│                │ (bedside      │
 └─────────────┘                └──────┬──────┘                │  manner)      │
   baselines +                          │                       └───────┬───────┘
   pattern buffer              ChromaDB: episodic memory              │
                               + medical knowledge base        Living Profile
                                                               (tone adapts)
```

1. **Profiler** — the silent watcher. Compares packets to the user's baselines and tracks deviations in a pattern buffer. Once a pattern is confirmed (e.g. 3 consecutive elevated-stress packets) it hands off.
2. **Analyst** — the doctor. Queries episodic memory and the medical knowledge base (RAG), then assigns a severity level (1–5) and drafts the facts.
3. **Communicator** — the bedside manner. Rewrites the Analyst's facts using the Living Profile ("warm & friendly" vs "brief & direct") and handles all direct chat.

## 🎮 Using the app

1. **Pick a scenario** on the landing page:
   - 👤 **Alex, 32** — chronic stress: sleep, HRV and recovery slowly deteriorate.
   - 👵 **Eleanor, 72** — acute emergency: hypertension + type 2 diabetes, sudden cardiac-style event.
2. **Stream the vitals** with **▶ Play** (one packet every 2 seconds), or **⏩ Skip to end** to process everything at once. Pause and resume any time.
3. **Watch the dashboard:** vital cards update against baseline, history charts fill in, and AVA messages you when a confirmed pattern appears.
4. **Chat with AVA** — and watch it adapt:

   | Try saying… | AVA learns… |
   |---|---|
   | "ok", "sure", "fine" | you're being dismissive → shorter, more engaging replies |
   | "what's my HRV?", "show me the numbers" | you like data |
   | "keep it brief", "cut to the chase" | shorter, more direct |
   | "be warmer", "be friendly" | a warmer tone |
   | "explain in detail", "elaborate" | longer, deeper answers |

5. **Self-report** how you feel from the sidebar (e.g. "started new meds") and AVA factors it in.

> **Note:** AVA is a technology demo running on simulated data. It is not a medical device and does not give medical advice.

## 💻 Run locally

**Prerequisites:** Python 3.10+ and a free [Groq API key](https://console.groq.com/).

```bash
# 1. Clone
git clone https://github.com/yashraj-a11y/adaptive-health-agent.git
cd adaptive-health-agent/adaptive_health_agent

# 2. Virtual environment + dependencies
python3 -m venv streamlit_env
source streamlit_env/bin/activate        # Windows: streamlit_env\Scripts\activate
pip install -r requirements.txt

# 3. API keys
cp .env.example .env                     # then edit .env and paste your key(s)

# 4. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>. The knowledge base and vector store are built automatically on first launch.

## ☁️ Deploy to Streamlit Cloud

The app is deployed from this repo at **[ava-healthagent.streamlit.app](https://ava-healthagent.streamlit.app/)**. To deploy your own copy:

1. Fork the repo and create a new app on [share.streamlit.io](https://share.streamlit.io/) with **Main file path** `adaptive_health_agent/app.py`.
2. In **Settings → Secrets**, add your keys:
   ```toml
   GROQ_API_KEY_PROFILER = "gsk_..."
   GROQ_API_KEY_ANALYST = "gsk_..."
   GROQ_API_KEY_COMMUNICATOR = "gsk_..."
   ```

Embeddings use ChromaDB's built-in ONNX MiniLM model rather than PyTorch, which keeps the install small enough for the free tier.

## ⚙️ Configuration

Copy `.env.example` to `.env`. Only the Groq keys are required; the rest are optional tuning. The three Groq keys can be the same key, or separate keys to spread rate limits across agents.

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY_PROFILER` / `_ANALYST` / `_COMMUNICATOR` | Groq key for each agent |
| `CHROMA_DB_PATH` | Where the vector store lives (default `./chroma_db`) |
| `MESSAGE_COOLDOWN_MINUTES` | Minimum gap between proactive messages from the Analyst (default `120`) |
| `SUMMARIZATION_INTERVAL_DAYS` | Window used when compressing episodic memory (default `7`) |

## 📂 Project structure

```text
adaptive_health_agent/
├── agents/
│   ├── profiler.py          # Monitors packets & pattern buffer
│   ├── analyst.py           # Assigns severity & context via RAG
│   └── communicator.py      # Adapts tone & handles user chat
├── graph/
│   ├── graph.py             # LangGraph state machine linking the agents
│   └── state.py             # Shared graph state
├── memory/
│   ├── living_profile.py    # User state (baselines, communication style)
│   ├── episodic_memory.py   # ChromaDB interactions
│   └── summarizer.py        # Memory compression
├── knowledge_base/          # Medical knowledge documents + loader
├── telemetry/
│   ├── user_a_scenario.py   # Mock data generator (stress)
│   └── user_b_scenario.py   # Mock data generator (emergency)
├── utils/
│   ├── baseline_calculator.py
│   └── pattern_buffer.py    # Consecutive-anomaly confirmation
├── tests/                   # Phase-by-phase tests
└── app.py                   # Streamlit UI
```
