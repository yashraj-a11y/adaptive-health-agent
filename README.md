# 🫀 AVA (Adaptive Virtual Assistant)

A completely dynamic, LLM-powered wearable health advisor. Instead of relying on static dashboards and simple alerts, AVA features a **3-agent architecture** that simulates a next-generation smartwatch experience: it monitors live vitals, learns your baselines, detects long-term patterns, and proactively talks to you in a tone tailored specifically to your personality.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-green.svg)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector-orange.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1_8B-black.svg)

---

## 🌟 Key Features

* **Live Telemetry Stream:** Simulates a wearable data stream (Heart Rate, HRV, SpO₂, Stress, Recovery, Sleep) feeding into the agent in real-time.
* **The Living Profile:** A LangGraph state object that tracks your physical baselines and dynamically updates your *communication profile* (directness, length, tone) based on how you reply to the agent in chat.
* **Episodic Memory (RAG):** Powered by **ChromaDB**. Anomalies and conversations are stored as memories, allowing the agent to recall past events ("You had a similar stress spike last week").
* **Smart Cost / Rate Limiting:** Avoids LLM calls on every packet. The system uses lightweight math to detect deviations and a "Pattern Buffer" (requiring 3 consecutive anomalies) before waking up the LLMs.
* **Sleek Split-Screen UI:** Built in Streamlit. Left side: Chat interface. Right side: Live streaming vitals panel.

## 🧠 The 3-Agent Architecture

1. **Profiler:** The silent watcher. Analyzes incoming telemetry packets against your established baselines. Tracks deviations using a pattern buffer. Once a pattern is confirmed (e.g., 3 consecutive elevated stress packets), it hands off to the Analyst.
2. **Analyst:** The doctor. Takes the confirmed pattern, queries the episodic memory (ChromaDB) and medical knowledge base, and assigns a Severity Level (1-5). It drafts the facts and passes them to the Communicator.
3. **Communicator:** The bedside manner. Takes the Analyst's facts and rewrites them based on your Living Profile (e.g., "Warm & Friendly" vs "Brief & Direct"). Handles all direct chat interaction with the user.

## 🚀 How to Run Locally

### 1. Clone & Setup
```bash
git clone <your-repo-url>
cd HyperPersonalised_WearableSmartWatch_AGENT

# Setup virtual environment
python3 -m venv streamlit_env
source streamlit_env/bin/activate
```

### 2. Install Dependencies
Ensure you have the required packages. (If no `requirements.txt` is present, install the core libraries):
```bash
pip install streamlit langgraph langchain chromadb sentence-transformers groq python-dotenv
```

### 3. Environment Variables
Create a `.env` file inside the `adaptive_health_agent` directory and add your Groq API keys. (The system supports splitting API limits across three keys, but you can use the same key for all three if desired).
```env
GROQ_API_KEY_PROFILER=gsk_your_key_here
GROQ_API_KEY_ANALYST=gsk_your_key_here
GROQ_API_KEY_COMMUNICATOR=gsk_your_key_here
```

### 4. Run the Application
```bash
cd adaptive_health_agent
streamlit run app.py
```

## 🎮 How to Use the App

1. **Select a Scenario:** Upon launching, choose between **Alex** (stress accumulation scenario) or **Eleanor** (medical emergency scenario).
2. **Initialize:** Click **Initialize / Reset** to populate the Living Profile and ChromaDB state.
3. **Stream Vitals:** Click **▶ Start** in the sidebar to begin processing the live telemetry mock feed (1 packet every 2 seconds).
4. **Chat:** Ask the agent questions, or simply wait for it to proactively reach out when it detects a confirmed anomaly. Try being dismissive ("ok", "whatever") or asking for data ("what is my HRV?") to see the agent adapt its tone!

## 📂 Project Structure

```text
adaptive_health_agent/
├── agents/
│   ├── profiler.py        # Monitors packets & pattern buffer
│   ├── analyst.py         # Assigns severity & context via RAG
│   └── communicator.py    # Adapts tone & handles user chat
├── graph/
│   └── graph.py           # LangGraph state machine linking agents
├── memory/
│   ├── living_profile.py  # User state management (baselines, style)
│   ├── episodic_memory.py # ChromaDB interactions
│   └── summarizer.py      # Background task for memory compression
├── telemetry/
│   ├── user_a_scenario.py # Mock data generator (Stress)
│   └── user_b_scenario.py # Mock data generator (Emergency)
├── utils/
│   └── pattern_buffer.py  # Cooldown logic for telemetry
└── app.py                 # Streamlit UI
```
