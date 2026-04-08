---
title: Email Triage OpenEnv
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# 📬 Email Triage OpenEnv

An **OpenEnv-compliant RL environment** for training and evaluating AI agents on real-world email triage, it is one of the most universally practiced business tasks, yet rarely modeled for agent evaluation.

---

## Why Email Triage?

Every professional triages hundreds of emails per week. Poor triage means missed crises, regulatory failures, damaged relationships. A capable agent must:

- Distinguish urgent crises from noise under time pressure
- Write empathetic, professional replies that address actual concerns
- Route correctly across legal, billing, HR, ops, PR, security teams
- Resist obvious spam while not mis-classifying legitimate external mail

This makes email triage a rich, multi-dimensional evaluation surface for LLMs.

---

## Environment Description

The environment presents a queue of realistic business emails one at a time. For each email, the agent must output:

| Field   | Description |
|---------|-------------|
| `label` | One of `urgent`, `respond`, `archive`, `spam`, `delegate` |
| `reply` | Optional reply text (required for `respond`, recommended for `urgent`) |
| `flag`  | Optional routing tag: `legal`, `billing`, `hr`, `marketing`, `ops`, `security`, `pr` |

---

## Action & Observation Space

### Action: `TriageAction`

```python
class TriageAction(BaseModel):
    label: str            # "urgent" | "respond" | "archive" | "spam" | "delegate"
    reply: Optional[str]  # reply text
    flag:  Optional[str]  # routing flag
```

### Observation: `TriageObservation`

```python
class TriageObservation(BaseModel):
    current_email:        Optional[EmailItem]  # email to triage
    emails_remaining:     int                  # queue depth
    last_action_feedback: str                  # grader feedback
    score_so_far:         float                # running normalized score [0,1]
    done:                 bool
```

### State: `TriageState`

```python
class TriageState(BaseModel):
    episode_id:           Optional[str]
    step_count:           int
    task_name:            str
    total_emails:         int
    total_reward:         float
    max_possible_reward:  float
```

---

## Tasks

| Task | Difficulty | Emails | Description |
|------|-----------|--------|-------------|
| `basic_triage`    | ⭐ Easy   | 5  | Obvious spam, clear urgent, newsletters, simple reply |
| `priority_inbox`  | ⭐⭐ Medium | 8  | Legal notices, billing, PR, HR — needs correct flag |
| `crisis_response` | ⭐⭐⭐ Hard | 10 | Simultaneous crises, regulator notices, media inquiries — reply quality graded |

### Reward Function

Each email's reward is a weighted sum of three dimensions:

```
reward = label_score × w_label + reply_score × w_reply + flag_score × w_flag
```

- **Label score**: 1.0 correct, 0.3 for close miss (e.g. urgent→respond), 0.0 otherwise
- **Reply score**: keyword coverage ratio + length bonus (0.0–1.0)
- **Flag score**: 1.0 correct flag, 0.0 wrong or missing when expected

This provides **dense, partial-progress signal** rather than sparse episode-end rewards.

### Difficulty progression

- **Easy**: All reward on label only (clear-cut classifications)
- **Medium**: 50% label, 50% flag (correct routing matters)
- **Hard**: 30% label + 40% reply quality + 30% flag (all three dimensions matter)

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- `pip install openenv-core httpx pydantic fastapi uvicorn openai`

### Run locally

```bash
# Start the server
cd email_triage_env
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline
export HF_TOKEN=your_token
export SERVER_URL=http://localhost:8000
python inference.py
```

### Run with Docker

```bash
# Build
docker build -t email-triage-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 email-triage-env:latest

# Run inference (in another terminal)
export HF_TOKEN=your_token
export SERVER_URL=http://localhost:8000
export IMAGE_NAME=email-triage-env:latest
python inference.py
```

### Validate

```bash
pip install openenv-core
openenv validate
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start episode. Body: `{"task_name": "basic_triage"}` |
| `/step` | POST | Take action. Body: `{"action": {...}}` |
| `/state` | GET | Episode metadata |
| `/tasks` | GET | List valid task names |

---

## Baseline Scores

Run with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score |
|------|-------|
| `basic_triage` | ~0.82 |
| `priority_inbox` | ~0.61 |
| `crisis_response` | ~0.44 |
| **Overall** | **~0.62** |

---

## Project Structure

```
email_triage_env/
├── __init__.py              # Public API exports
├── models.py                # Pydantic Action/Observation/State
├── client.py                # Async + sync client
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py               # FastAPI application
    ├── email_triage_environment.py  # Core logic + graders
    ├── requirements.txt
    └── Dockerfile
```

---

## License

MIT