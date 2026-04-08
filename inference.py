"""
Inference Script — Email Triage OpenEnv
========================================
MANDATORY environment variables:
  API_BASE_URL   – LLM API endpoint  (default: HuggingFace router)
  MODEL_NAME     – Model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       – HuggingFace / API key
  IMAGE_NAME     – Docker image name for the env (optional if server already running)

STDOUT FORMAT (strict):
  [START] task=<name> env=email_triage model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Run:
  HF_TOKEN=<token> IMAGE_NAME=email-triage-env:latest python inference.py
"""

import asyncio
import json
import os
import textwrap
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME", "email-triage-env:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8000")

BENCHMARK = "email_triage"
MAX_STEPS = 12           # safety cap (actual episodes are 5/8/10 emails)
TEMPERATURE = 0.3
MAX_TOKENS = 400
SUCCESS_SCORE_THRESHOLD = 0.5   # normalized score ≥ 0.5 → success

TASKS = ["basic_triage", "priority_inbox", "crisis_response"]


# ─────────────────────────────────────────────
# Stdout loggers
# ─────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Sanitize action string (no newlines)
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean!r} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# System & user prompts
# ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert email triage assistant for a busy executive.
For each email shown, you must decide how to handle it.

LABELS (choose exactly one):
  urgent   – needs immediate attention (crisis, VIP, deadline < 24h)
  respond  – requires a thoughtful reply from you
  delegate – forward to the right team (billing, hr, legal, marketing, ops, security, pr)
  archive  – informational, newsletters, internal updates — no action needed
  spam     – unsolicited marketing, scams, irrelevant

ROUTING FLAGS (use when label is urgent or delegate):
  legal | billing | hr | marketing | ops | security | pr

REPLY (required when label is "respond", highly recommended for "urgent"):
  Write a concise, professional reply. Include relevant keywords:
  - For "urgent" crises: mention investigation, timeline, reassurance.
  - For "respond" requests: be helpful and specific.

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown fences:
{
  "label": "<one of the 5 labels>",
  "reply": "<reply text or null>",
  "flag": "<routing flag or null>"
}
""").strip()


def build_user_prompt(email: dict, step: int, score_so_far: float, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    email_block = textwrap.dedent(f"""
    FROM:    {email['sender']}
    SUBJECT: {email['subject']}
    BODY:
    {email['body']}
    """).strip()

    return textwrap.dedent(f"""
    Step {step} | Running score: {score_so_far:.2%}

    --- EMAIL ---
    {email_block}

    --- RECENT ACTIONS ---
    {history_block}

    Triage this email now. Reply with JSON only.
    """).strip()


# ─────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────

def get_model_action(
    client: OpenAI,
    email: dict,
    step: int,
    score_so_far: float,
    history: List[str],
) -> dict:
    """Call the LLM and parse its JSON triage decision."""
    prompt = build_user_prompt(email, step, score_so_far, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        label = str(parsed.get("label", "archive")).lower()
        reply = parsed.get("reply") or None
        flag = parsed.get("flag") or None
        return {"label": label, "reply": reply, "flag": flag}
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return {"label": "archive", "reply": None, "flag": None}


# ─────────────────────────────────────────────
# HTTP helpers (no SDK — raw httpx for portability)
# ─────────────────────────────────────────────

def http_reset(task_name: str, timeout: float = 30.0) -> dict:
    r = httpx.post(f"{SERVER_URL}/reset", json={"task_name": task_name}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def http_step(action: dict, timeout: float = 30.0) -> dict:
    r = httpx.post(f"{SERVER_URL}/step", json={"action": action}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def wait_for_server(retries: int = 30, delay: float = 1.5) -> bool:
    for _ in range(retries):
        try:
            r = httpx.get(f"{SERVER_URL}/health", timeout=3.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


# ─────────────────────────────────────────────
# Single-task runner
# ─────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> float:
    """
    Run one full episode for `task_name`.
    Emits [START], [STEP]*, [END] lines.
    Returns final normalized score.
    """
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    try:
        obs_data = http_reset(task_name)
        score_so_far = obs_data.get("score_so_far", 0.0)

        for step in range(1, MAX_STEPS + 1):
            done = obs_data.get("done", False)
            if done:
                break

            current_email = obs_data.get("current_email")
            if current_email is None:
                break

            # Get model decision
            action = get_model_action(client, current_email, step, score_so_far, history)
            action_str = f"label={action['label']},flag={action['flag']},reply_len={len(action.get('reply') or '')}"

            # Step the environment
            try:
                step_data = http_step(action)
                reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)
                info = step_data.get("info", {})
                obs_data = step_data.get("observation", {})
                score_so_far = obs_data.get("score_so_far", 0.0)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)[:100]

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: label={action['label']} flag={action['flag']} → reward {reward:+.2f}"
            )

            if done or error:
                break

        # Compute final score
        total_emails = max(1, len(rewards))
        score = sum(rewards) / total_emails
        score = round(min(max(score, 0.0), 1.0), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task '{task_name}' error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Wait for server to be ready
    print(f"[DEBUG] Waiting for server at {SERVER_URL} ...", flush=True)
    if not wait_for_server():
        print("[DEBUG] Server not reachable. Exiting.", flush=True)
        # Still emit valid [END] lines for each task
        for task in TASKS:
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    all_scores = []
    for task in TASKS:
        score = run_task(client, task)
        all_scores.append(score)
        print(f"[DEBUG] Task '{task}' score: {score:.3f}", flush=True)

    overall = sum(all_scores) / len(all_scores)
    print(f"[DEBUG] Overall average score: {overall:.3f}", flush=True)


if __name__ == "__main__":
    main()