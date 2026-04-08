"""
Email Triage Environment — core logic.

Three tasks of increasing difficulty:
  1. basic_triage    (easy)   – 5 emails, obvious spam + urgent labels
  2. priority_inbox  (medium) – 8 emails, nuanced urgent vs respond, requires correct flag
  3. crisis_response (hard)   – 10 emails, ambiguous signals, reply quality graded,
                                time-pressure metadata, delegation chains
"""

import uuid
import re
from typing import List, Optional, Dict, Tuple
from copy import deepcopy

try:
    from models import TriageAction, TriageObservation, TriageState, EmailItem
except ImportError:
    from email_triage_env.models import TriageAction, TriageObservation, TriageState, EmailItem


# ─────────────────────────────────────────────────────────────
# Email datasets per task
# ─────────────────────────────────────────────────────────────

TASK_EMAILS: Dict[str, List[dict]] = {

    "basic_triage": [
        {
            "id": "e001",
            "sender": "noreply@sweepstakes99.biz",
            "subject": "YOU WON $5000!!!",
            "body": "Congratulations! Click the link below to claim your prize. Limited time only!",
            "timestamp": "2024-03-01T08:00:00Z",
            "metadata": {},
            "correct_label": "spam",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
        },
        {
            "id": "e002",
            "sender": "ceo@company.com",
            "subject": "URGENT: Production server is down",
            "body": "The main API server went offline 10 minutes ago. Customers cannot access the platform. We need a response NOW.",
            "timestamp": "2024-03-01T08:05:00Z",
            "metadata": {"from_vip": True},
            "correct_label": "urgent",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
        },
        {
            "id": "e003",
            "sender": "newsletter@techblog.com",
            "subject": "This week in AI - March 2024",
            "body": "Top stories: LLMs get smarter, robotics breakthroughs, and more.",
            "timestamp": "2024-03-01T08:10:00Z",
            "metadata": {},
            "correct_label": "archive",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
        },
        {
            "id": "e004",
            "sender": "customer@bigclient.com",
            "subject": "Question about our contract renewal",
            "body": "Hi, our contract is up next month. Could you let us know the renewal process and any updated pricing? Thanks.",
            "timestamp": "2024-03-01T08:15:00Z",
            "metadata": {},
            "correct_label": "respond",
            "correct_flag": None,
            "reward_label": 0.5,
            "reward_reply": 0.5,  # reply quality matters
        },
        {
            "id": "e005",
            "sender": "hr@company.com",
            "subject": "Reminder: Time-sheet submission due today",
            "body": "Please submit your time-sheets by 5 PM today. Contact HR if you have any issues.",
            "timestamp": "2024-03-01T08:20:00Z",
            "metadata": {},
            "correct_label": "archive",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
        },
    ],

    "priority_inbox": [
        {
            "id": "p001",
            "sender": "legal@partnerlaw.com",
            "subject": "Contract dispute — response required within 48h",
            "body": "We represent XYZ Corp and are contacting you regarding clause 7.3 of the signed agreement dated Jan 2024. Failure to respond may result in escalation.",
            "timestamp": "2024-03-02T07:00:00Z",
            "metadata": {"deadline_hours": 48},
            "correct_label": "urgent",
            "correct_flag": "legal",
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
        {
            "id": "p002",
            "sender": "promo@shoestore.com",
            "subject": "50% off this weekend only!",
            "body": "Don't miss our biggest sale. Shop now!",
            "timestamp": "2024-03-02T07:05:00Z",
            "metadata": {},
            "correct_label": "spam",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
            "reward_flag": 0.0,
        },
        {
            "id": "p003",
            "sender": "billing@cloudprovider.com",
            "subject": "Invoice #4421 – $12,340.00 due",
            "body": "Your invoice for cloud services in February is attached. Payment due in 15 days.",
            "timestamp": "2024-03-02T07:10:00Z",
            "metadata": {},
            "correct_label": "delegate",
            "correct_flag": "billing",
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
        {
            "id": "p004",
            "sender": "reporter@techcrunch.com",
            "subject": "Comment request for story on your funding round",
            "body": "I'm writing a piece about your Series B. Would love a quote from your CEO by tomorrow morning.",
            "timestamp": "2024-03-02T07:15:00Z",
            "metadata": {"deadline_hours": 18},
            "correct_label": "urgent",
            "correct_flag": "pr",
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
        {
            "id": "p005",
            "sender": "candidate@gmail.com",
            "subject": "Follow-up on my software engineer application",
            "body": "I applied 3 weeks ago and haven't heard back. Could you provide an update?",
            "timestamp": "2024-03-02T07:20:00Z",
            "metadata": {},
            "correct_label": "delegate",
            "correct_flag": "hr",
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
        {
            "id": "p006",
            "sender": "partner@bigretail.com",
            "subject": "Proposal for joint marketing campaign",
            "body": "We'd like to propose a co-branded campaign for Q3. Would you be interested in a call next week?",
            "timestamp": "2024-03-02T07:25:00Z",
            "metadata": {},
            "correct_label": "respond",
            "correct_flag": None,
            "reward_label": 0.5,
            "reward_reply": 0.5,
            "reward_flag": 0.0,
        },
        {
            "id": "p007",
            "sender": "devops@company.com",
            "subject": "Scheduled maintenance this Saturday 2-4 AM",
            "body": "We will be performing DB maintenance. No customer-visible downtime expected.",
            "timestamp": "2024-03-02T07:30:00Z",
            "metadata": {},
            "correct_label": "archive",
            "correct_flag": None,
            "reward_label": 1.0,
            "reward_reply": 0.0,
            "reward_flag": 0.0,
        },
        {
            "id": "p008",
            "sender": "security@company.com",
            "subject": "ALERT: Unusual login detected on your account",
            "body": "We detected a login from an unrecognized device in Russia. If this wasn't you, change your password immediately.",
            "timestamp": "2024-03-02T07:35:00Z",
            "metadata": {"from_vip": True},
            "correct_label": "urgent",
            "correct_flag": "security",
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
    ],

    "crisis_response": [
        {
            "id": "c001",
            "sender": "customer_support@bigbank.com",
            "subject": "Fraudulent transactions on client account",
            "body": "Multiple unauthorized transactions were flagged on account #938471. Client is demanding immediate action. Escalating to you as the account manager.",
            "timestamp": "2024-03-03T06:00:00Z",
            "metadata": {"escalated": True, "deadline_hours": 2},
            "correct_label": "urgent",
            "correct_flag": "legal",
            "reply_keywords": ["investigating", "24 hours", "apologize", "secure", "immediate"],
            "reward_label": 0.3,
            "reward_reply": 0.4,
            "reward_flag": 0.3,
        },
        {
            "id": "c002",
            "sender": "influencer@socialmedia.com",
            "subject": "Collab opportunity — 2M followers",
            "body": "Hey! I'd love to partner with you for a sponsored post. Let me know if you're interested!",
            "timestamp": "2024-03-03T06:05:00Z",
            "metadata": {},
            "correct_label": "delegate",
            "correct_flag": "marketing",
            "reply_keywords": [],
            "reward_label": 0.5,
            "reward_reply": 0.0,
            "reward_flag": 0.5,
        },
        {
            "id": "c003",
            "sender": "regulator@sec.gov",
            "subject": "Inquiry Notice — Ref #SEC-2024-0391",
            "body": "This is a formal notice of inquiry regarding your firm's Q4 reporting practices. You are required to provide documentation within 10 business days.",
            "timestamp": "2024-03-03T06:10:00Z",
            "metadata": {"deadline_hours": 240, "from_vip": True},
            "correct_label": "urgent",
            "correct_flag": "legal",
            "reply_keywords": ["acknowledge", "cooperate", "counsel", "documentation"],
            "reward_label": 0.3,
            "reward_reply": 0.4,
            "reward_flag": 0.3,
        },
        {
            "id": "c004",
            "sender": "employee@company.com",
            "subject": "I want to report a workplace incident",
            "body": "I witnessed inappropriate behavior from a manager last week. I'm not sure who to tell. Is this confidential?",
            "timestamp": "2024-03-03T06:15:00Z",
            "metadata": {},
            "correct_label": "delegate",
            "correct_flag": "hr",
            "reply_keywords": ["confidential", "safe", "hr", "support"],
            "reward_label": 0.3,
            "reward_reply": 0.4,
            "reward_flag": 0.3,
        },
        {
            "id": "c005",
            "sender": "datacenter@hostprovider.com",
            "subject": "Critical: SSD failure in rack C-12",
            "body": "RAID array has lost 2 drives in rack C-12. Redundancy is compromised. Immediate action required to prevent data loss.",
            "timestamp": "2024-03-03T06:20:00Z",
            "metadata": {"deadline_hours": 1, "escalated": True},
            "correct_label": "urgent",
            "correct_flag": "ops",
            "reply_keywords": ["immediately", "backup", "restore", "team"],
            "reward_label": 0.4,
            "reward_reply": 0.3,
            "reward_flag": 0.3,
        },
        {
            "id": "c006",
            "sender": "noreply@discount-pharma.net",
            "subject": "Get your meds cheap — no prescription needed",
            "body": "Order any medication without a prescription. Discreet shipping. Visit our website.",
            "timestamp": "2024-03-03T06:25:00Z",
            "metadata": {},
            "correct_label": "spam",
            "correct_flag": None,
            "reply_keywords": [],
            "reward_label": 1.0,
            "reward_reply": 0.0,
            "reward_flag": 0.0,
        },
        {
            "id": "c007",
            "sender": "journalist@guardian.com",
            "subject": "Right of reply: data breach allegations",
            "body": "We are publishing a story tomorrow about an alleged data breach at your company. Do you wish to comment? Deadline: 6 PM today.",
            "timestamp": "2024-03-03T06:30:00Z",
            "metadata": {"deadline_hours": 12, "from_vip": True},
            "correct_label": "urgent",
            "correct_flag": "pr",
            "reply_keywords": ["investigating", "comment", "statement", "pr", "legal", "privacy"],
            "reward_label": 0.3,
            "reward_reply": 0.4,
            "reward_flag": 0.3,
        },
        {
            "id": "c008",
            "sender": "board@company.com",
            "subject": "Emergency board call — today 3 PM",
            "body": "Given recent events, an emergency board meeting has been called for 3 PM today. Your presence is mandatory.",
            "timestamp": "2024-03-03T06:35:00Z",
            "metadata": {"from_vip": True, "deadline_hours": 9},
            "correct_label": "urgent",
            "correct_flag": None,
            "reply_keywords": ["confirm", "attend", "present"],
            "reward_label": 0.5,
            "reward_reply": 0.5,
            "reward_flag": 0.0,
        },
        {
            "id": "c009",
            "sender": "newsletter@cookingrecipes.com",
            "subject": "5 delicious pasta recipes for the weekend",
            "body": "Try our new carbonara and lasagna recipes!",
            "timestamp": "2024-03-03T06:40:00Z",
            "metadata": {},
            "correct_label": "archive",
            "correct_flag": None,
            "reply_keywords": [],
            "reward_label": 1.0,
            "reward_reply": 0.0,
            "reward_flag": 0.0,
        },
        {
            "id": "c010",
            "sender": "vp_sales@company.com",
            "subject": "Q1 pipeline update — need your sign-off",
            "body": "Attached is the Q1 sales pipeline. We need your written approval before presenting to the board today at 3 PM.",
            "timestamp": "2024-03-03T06:45:00Z",
            "metadata": {"deadline_hours": 9, "from_vip": True},
            "correct_label": "urgent",
            "correct_flag": None,
            "reply_keywords": ["approve", "confirmed", "sign", "reviewed"],
            "reward_label": 0.5,
            "reward_reply": 0.5,
            "reward_flag": 0.0,
        },
    ],
}


# ─────────────────────────────────────────────────────────────
# Grader helpers
# ─────────────────────────────────────────────────────────────

def _grade_reply(reply_text: Optional[str], keywords: List[str]) -> float:
    """Score reply quality 0.0–1.0 based on keyword coverage."""
    if not keywords:
        return 1.0  # no reply needed — full score on that dimension
    if not reply_text or len(reply_text.strip()) < 10:
        return 0.0
    lower = reply_text.lower()
    hit = sum(1 for kw in keywords if kw.lower() in lower)
    # partial credit + length bonus
    base = hit / len(keywords)
    length_bonus = min(0.1, len(reply_text) / 500)
    return min(1.0, base + length_bonus)


def _grade_step(email_spec: dict, action: TriageAction) -> Tuple[float, str]:
    """
    Returns (reward, feedback_string) for a single triage step.
    reward is in [0.0, 1.0] relative to the weights in the email spec.
    """
    correct_label = email_spec["correct_label"]
    correct_flag = email_spec.get("correct_flag")
    reward_label_w = email_spec.get("reward_label", 1.0)
    reward_reply_w = email_spec.get("reward_reply", 0.0)
    reward_flag_w = email_spec.get("reward_flag", 0.0)
    reply_keywords = email_spec.get("reply_keywords", [])

    label_score = 1.0 if action.label == correct_label else 0.0

    # Partial credit for close misses on respond/urgent
    if label_score == 0.0:
        close_pairs = {("urgent", "respond"), ("respond", "urgent"), ("delegate", "respond")}
        if (action.label, correct_label) in close_pairs:
            label_score = 0.3

    reply_score = _grade_reply(action.reply, reply_keywords) if reward_reply_w > 0 else 1.0

    flag_score = 0.0
    if reward_flag_w > 0:
        if correct_flag is None:
            flag_score = 1.0 if action.flag is None else 0.8  # minor penalty for unnecessary flag
        else:
            if action.flag and action.flag.lower() == correct_flag.lower():
                flag_score = 1.0
            elif action.flag:
                flag_score = 0.0  # wrong flag
            else:
                flag_score = 0.0  # missing flag

    reward = (
        label_score * reward_label_w
        + reply_score * reward_reply_w
        + flag_score * reward_flag_w
    )
    reward = round(min(1.0, max(0.0, reward)), 4)

    # Build feedback
    parts = []
    if label_score == 1.0:
        parts.append(f"✓ Label '{action.label}' correct.")
    elif label_score > 0:
        parts.append(f"~ Label '{action.label}' partially correct (expected '{correct_label}').")
    else:
        parts.append(f"✗ Label '{action.label}' incorrect (expected '{correct_label}').")

    if reward_reply_w > 0:
        if reply_score >= 0.8:
            parts.append(f"✓ Reply quality good ({reply_score:.0%}).")
        elif reply_score >= 0.4:
            parts.append(f"~ Reply quality moderate ({reply_score:.0%}).")
        else:
            parts.append(f"✗ Reply missing or low quality ({reply_score:.0%}).")

    if reward_flag_w > 0:
        if flag_score == 1.0:
            parts.append(f"✓ Flag '{action.flag}' correct.")
        else:
            parts.append(f"✗ Flag incorrect (expected '{correct_flag}', got '{action.flag}').")

    feedback = " ".join(parts) + f" Step reward: {reward:.2f}"
    return reward, feedback


# ─────────────────────────────────────────────────────────────
# Environment class
# ─────────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    Email Triage OpenEnv Environment.

    Episode flow:
      reset(task_name) → initial observation with first email
      step(action)     → observation with next email + reward
      state()          → episode metadata
    """

    VALID_TASKS = list(TASK_EMAILS.keys())

    def __init__(self):
        self._task_name: str = "basic_triage"
        self._queue: List[dict] = []
        self._current_idx: int = 0
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._max_reward: float = 0.0
        self._done: bool = False
        self._last_feedback: str = "Episode not started."
        self._score_so_far: float = 0.0

    # ── Internal helpers ──────────────────────────────────────

    def _build_observation(self) -> TriageObservation:
        if self._done or self._current_idx >= len(self._queue):
            return TriageObservation(
                current_email=None,
                emails_remaining=0,
                last_action_feedback=self._last_feedback,
                score_so_far=round(self._score_so_far, 4),
                done=True,
            )
        spec = self._queue[self._current_idx]
        email = EmailItem(
            id=spec["id"],
            sender=spec["sender"],
            subject=spec["subject"],
            body=spec["body"],
            timestamp=spec["timestamp"],
            metadata=spec.get("metadata", {}),
        )
        remaining = len(self._queue) - self._current_idx - 1
        return TriageObservation(
            current_email=email,
            emails_remaining=remaining,
            last_action_feedback=self._last_feedback,
            score_so_far=round(self._score_so_far, 4),
            done=False,
        )

    # ── Public API ────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> TriageObservation:
        if task_name is None:
            task_name = "basic_triage"
        if task_name not in self.VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {self.VALID_TASKS}")

        self._task_name = task_name
        self._queue = deepcopy(TASK_EMAILS[task_name])
        self._current_idx = 0
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        self._max_reward = float(len(self._queue))  # max 1.0 per email
        self._done = False
        self._last_feedback = f"Task '{task_name}' started. {len(self._queue)} emails to triage."
        self._score_so_far = 0.0

        return self._build_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already done"}

        if self._current_idx >= len(self._queue):
            self._done = True
            obs = self._build_observation()
            return obs, 0.0, True, {}

        spec = self._queue[self._current_idx]
        reward, feedback = _grade_step(spec, action)

        self._total_reward += reward
        self._step_count += 1
        self._last_feedback = feedback
        self._score_so_far = self._total_reward / max(1, self._max_reward)
        self._current_idx += 1

        if self._current_idx >= len(self._queue):
            self._done = True
            self._last_feedback += f" | Episode complete. Final score: {self._score_so_far:.2f}"

        obs = self._build_observation()
        return obs, reward, self._done, {"feedback": feedback}

    def state(self) -> TriageState:
        return TriageState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._task_name,
            total_emails=len(self._queue),
            correct_labels=0,  # tracked via reward
            total_reward=round(self._total_reward, 4),
            max_possible_reward=round(self._max_reward, 4),
        )