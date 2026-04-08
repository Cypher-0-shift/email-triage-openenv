"""
Pydantic models for the Email Triage Environment.

Action:      TriageAction  - label + optional_reply + optional_flag
Observation: TriageObservation - current email + queue status + feedback
State:       TriageState - episode metadata
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    Agent's decision on the current email.

    Fields:
      label:   One of "urgent", "respond", "archive", "spam", "delegate"
      reply:   Optional reply text (required when label == "respond")
      flag:    Optional flag/tag for downstream routing, e.g. "legal", "billing"
    """
    label: str = Field(
        description='Triage decision: "urgent" | "respond" | "archive" | "spam" | "delegate"'
    )
    reply: Optional[str] = Field(
        default=None,
        description="Reply text to send (required when label is 'respond')"
    )
    flag: Optional[str] = Field(
        default=None,
        description="Optional routing flag e.g. 'legal', 'billing', 'hr'"
    )


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class EmailItem(BaseModel):
    """A single email in the inbox."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TriageObservation(BaseModel):
    """
    What the agent sees after each step.

    Fields:
      current_email:   The email to triage right now (None if queue empty)
      emails_remaining: How many more emails are left in the queue
      last_action_feedback: Textual feedback on the previous action
      score_so_far:    Running normalized score [0.0, 1.0]
      done:            True if the episode is finished
    """
    current_email: Optional[EmailItem] = Field(
        default=None,
        description="The email that needs to be triaged"
    )
    emails_remaining: int = Field(
        default=0,
        description="Number of emails still in the queue after this one"
    )
    last_action_feedback: str = Field(
        default="",
        description="Human-readable feedback on the previous triage action"
    )
    score_so_far: float = Field(
        default=0.0,
        description="Cumulative normalized score so far in [0.0, 1.0]"
    )
    done: bool = Field(
        default=False,
        description="True when the episode has ended"
    )


# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class TriageState(BaseModel):
    """Episode-level metadata."""
    episode_id: Optional[str] = None
    step_count: int = 0
    task_name: str = ""
    total_emails: int = 0
    correct_labels: int = 0
    total_reward: float = 0.0
    max_possible_reward: float = 0.0