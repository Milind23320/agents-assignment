"""
Interrupt Handler for LiveKit Agents

Place this file in your agent project (e.g., livekit-plugins or livekit-agents utils)
and instantiate it when you create your AgentSession.

Purpose:
- Distinguish between passive "backchannels" ("yeah", "ok", "hmm") and active
  interruptions ("wait", "stop", "no").
- When the agent is SPEAKING, the handler will *ignore* short filler-only utterances
  (so the TTS doesn't stop). If the utterance contains any interrupt keywords,
  the agent will be stopped immediately.
- When the agent is SILENT, user input is handled normally (no filtering).

How it works (summary):
1. VAD fires quickly when the user utters something. Instead of immediately
   stopping TTS, we mark the VAD event as "pending" and subscribe to the
   STT stream for the same user.
2. We wait a very short time (configurable, default 150 ms) for the STT partial
   or final transcript. If the transcript contains only filler words, we ignore
   it when the agent is speaking. If it contains interrupt keywords, we interrupt.
3. If the STT doesn't arrive within the short window, we fall back to a safe
   default: treat the VAD as an interruption only if the agent is silent. If
   the agent is speaking we conservatively ignore (to prevent stuttering).

Notes on integration:
- This module is intentionally framework-agnostic. The real repo's AgentSession
  has methods/events such as `on_vad`, `on_stt_partial`, `on_stt_final`, and
  `stop_tts` or similar. You will need to connect the handler's callbacks to
  those event hooks. See the `integration_snippet()` function at the bottom for
  example wiring.

"""

import asyncio
import os
import re
import time
from typing import Callable, Dict, Iterable, List, Optional, Set

DEFAULT_IGNORE_WORDS = ["yeah", "ok", "hmm", "right", "uh-huh", "mm", "mhm"]
DEFAULT_INTERRUPT_WORDS = ["wait", "stop", "no", "hold on", "hold up", "pause"]

# Short helper: normalize transcripts
def normalize(text: str) -> str:
    text = text.lower().strip()
    # remove repeated punctuation and normalize whitespace
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    text = normalize(text)
    # check each keyword as standalone word or phrase
    for kw in keywords:
        kwn = normalize(kw)
        if not kwn:
            continue
        # word boundary search
        if re.search(rf"\b{re.escape(kwn)}\b", text):
            return True
    return False


class InterruptHandler:
    """Context-aware interrupt handler.

    Instantiate with a running `session` object from the LiveKit Agents framework.
    The handler listens for VAD and STT events and applies the ignore/interrupt
    logic described in the challenge.

    Important attributes / expected session interface (adapt as needed):
    - session.is_currently_speaking() -> bool
        Should return True when TTS audio is playing (or being generated) by the agent.
    - session.stop_tts_immediately() -> None
        Should immediately stop the TTS audio stream (if available).
    - session.handle_user_transcript(user_id, transcript, final=True) -> None
        Called to forward a validated transcript to the agent when we decide the
        user's speech should be processed.

    If the real session has different method names, adapt the integration snippet
    at the bottom of this file.
    """

    def __init__(
        self,
        session,
        ignore_words: Optional[Iterable[str]] = None,
        interrupt_words: Optional[Iterable[str]] = None,
        stt_wait_ms: int = 150,
        max_transcript_age_ms: int = 1500,
    ) -> None:
        self.session = session
        self.ignore_words: Set[str] = set(map(normalize, ignore_words or DEFAULT_IGNORE_WORDS))
        self.interrupt_words: Set[str] = set(map(normalize, interrupt_words or DEFAULT_INTERRUPT_WORDS))
        # how long to wait for the STT transcript to decide (short to avoid latency)
        self.stt_wait_ms = stt_wait_ms
        # drop transcripts older than this when matching VAD
        self.max_transcript_age_ms = max_transcript_age_ms

        # internal state
        # pending_vads: user_id -> { 'timestamp': float(ms), 'task': asyncio.Task }
        self.pending_vads: Dict[str, Dict] = {}
        # last transcripts per user: user_id -> (text, timestamp_ms, final)
        self.last_transcript: Dict[str, Dict] = {}

        # hooks (these can be replaced for testing)
        self._loop = asyncio.get_event_loop()

    # ------------------------- Event handlers -------------------------
    def on_vad_start(self, user_id: str) -> None:
        """Call this when VAD detects the user started speaking.

        We do *not* immediately stop audio. Instead we create a short delay and
        await the STT transcript (partial/final). Decision is based on combined
        rules and agent speaking state.
        """
        now = int(time.time() * 1000)
        # clear any previous pending for this user
        prev = self.pending_vads.get(user_id)
        if prev:
            # cancel previous decision task; we'll wait for a new transcript
            prev_task = prev.get("task")
            if prev_task and not prev_task.done():
                prev_task.cancel()

        task = self._loop.create_task(self._handle_pending_vad(user_id, now))
        self.pending_vads[user_id] = {"timestamp": now, "task": task}

    async def _handle_pending_vad(self, user_id: str, vad_ts_ms: int) -> None:
        """Wait a short window for STT; then decide what to do."""
        try:
            await asyncio.sleep(self.stt_wait_ms / 1000.0)
            # check for recent transcript
            tr = self.last_transcript.get(user_id)
            if tr:
                age = vad_ts_ms - tr["timestamp_ms"]
                # If transcript arrived BEFORE VAD or within allowed window, use it
                if abs(age) <= self.max_transcript_age_ms:
                    # Decide using transcript
                    self._decide_from_transcript(user_id, tr["text"], tr["final"])
                    return
            # No transcript or too old. Fallback policy:
            # - If agent is speaking -> ignore (prefer continuity)
            # - If agent is silent -> treat as interruption (so user isn't ignored)
            if self._agent_is_speaking():
                # nothing to do -> ignore VAD
                return
            else:
                # agent silent => treat as user input (forward to normal handler)
                # We can't supply a transcript, but we can notify the session to start
                # an STT session or mark that the user is speaking. Many frameworks
                # will already handle this; we leave it to session.
                if hasattr(self.session, "on_vad_fallback_no_transcript"):
                    # optional hook on session
                    self.session.on_vad_fallback_no_transcript(user_id)
                return
        except asyncio.CancelledError:
            return

    def on_stt_partial(self, user_id: str, text: str) -> None:
        """Record partial transcripts for quick decisions.

        Many STT providers emit partial results. We store the latest partial so
        when the VAD pending window expires we can decide quickly.
        """
        now = int(time.time() * 1000)
        self.last_transcript[user_id] = {"text": text, "timestamp_ms": now, "final": False}

    def on_stt_final(self, user_id: str, text: str) -> None:
        """Record final transcript and make an immediate decision if there's a
        pending VAD for this user (no need to wait the timeout)."""
        now = int(time.time() * 1000)
        self.last_transcript[user_id] = {"text": text, "timestamp_ms": now, "final": True}

        pending = self.pending_vads.get(user_id)
        if pending:
            # cancel the pending task and handle immediately
            task = pending.get("task")
            if task and not task.done():
                task.cancel()
            # run the decision synchronously (fire-and-forget)
            self._decide_from_transcript(user_id, text, final=True)

    # ------------------------- Decision logic -------------------------
    def _agent_is_speaking(self) -> bool:
        # adapt to session API; prefer an explicit method
        if hasattr(self.session, "is_currently_speaking"):
            try:
                return bool(self.session.is_currently_speaking())
            except Exception:
                pass
        # fallback heuristics
        for cand in ("tts_playing", "is_playing_audio", "tts_active"):
            if getattr(self.session, cand, False):
                return True
        return False

    def _stop_agent_audio(self) -> None:
        if hasattr(self.session, "stop_tts_immediately"):
            try:
                self.session.stop_tts_immediately()
                return
            except Exception:
                pass
        # fallback: try a common name
        if hasattr(self.session, "stop_audio"):
            try:
                self.session.stop_audio()
                return
            except Exception:
                pass
        # if we can't stop audio, raise a warning in logs (not raising exception)
        print("[InterruptHandler] Warning: couldn't call session stop method")

    def _decide_from_transcript(self, user_id: str, transcript: str, final: bool) -> None:
        """Core decision logic.

        Rules:
        - If agent speaking:
            * If transcript contains any interrupt word -> STOP immediately (interrupt).
            * Else if transcript contains only filler/ignore words (or is very short and matches ignore set) -> IGNORE.
            * If transcript contains both filler and interrupt keywords -> INTERRUPT (interrupt has priority).
        - If agent silent: forward transcript to session for normal handling.
        """
        text = normalize(transcript)
        if not text:
            return

        has_interrupt = contains_any_keyword(text, self.interrupt_words)
        has_filler = contains_any_keyword(text, self.ignore_words)

        # Priority: explicit interrupt words always win
        if self._agent_is_speaking():
            if has_interrupt:
                # immediate stop and forward transcript
                self._stop_agent_audio()
                # forward transcript to session to be processed as user input
                self._forward_transcript(user_id, transcript, final=final)
                return
            # no explicit interrupt. If transcript contains *only* filler words (or is short and matches filler), ignore
            # Decide if the transcript is filler-only: remove filler words and see if anything remains.
            tokens = [t for t in re.split(r"\s+", text) if t]
            non_filler_tokens = [t for t in tokens if normalize(t) not in self.ignore_words]
            if len(non_filler_tokens) == 0:
                # pure filler -> ignore entirely
                return
            # Mixed content like "yeah but wait" will have non-filler tokens; if any of those are interrupt words we'd already have returned.
            # For mixed partials (e.g., "yeah, I was thinking"), we treat as USER INTENT and forward (because user is saying something meaningful)
            # but keep continuity: we *do not* stop TTS for these — requirement: "If agent is speaking and user says a filler word, the agent must NOT stop."
            # So the rule is: if agent is speaking and the transcript contains non-filler but no explicit interrupt words, we should NOT stop audio — instead buffer the user's utterance and process after current utterance finishes.
            # Implementation choice: forward the transcript to session but do NOT stop audio. Session can decide to respond after TTS finishes.
            self._forward_transcript(user_id, transcript, final=final, stop_audio=False)
            return
        else:
            # Agent is silent: treat as normal user input
            self._forward_transcript(user_id, transcript, final=final)
            return

    # ------------------------- Forwarding -------------------------
    def _forward_transcript(self, user_id: str, transcript: str, final: bool = True, stop_audio: Optional[bool] = True) -> None:
        """Call into the session so the transcript is processed by the agent.

        stop_audio flag indicates whether we should stop audio before forwarding.
        """
        if stop_audio:
            # if stopping is desired and agent is speaking, ensure immediate stop
            if self._agent_is_speaking():
                self._stop_agent_audio()

        # prefer a canonical session API method
        if hasattr(self.session, "handle_user_transcript"):
            try:
                # synchronous call; if it's async, user should adapt when integrating
                maybe_coro = self.session.handle_user_transcript(user_id, transcript, final=final)
                if asyncio.iscoroutine(maybe_coro):
                    # schedule it
                    self._loop.create_task(maybe_coro)
                return
            except Exception as e:
                print(f"[InterruptHandler] session.handle_user_transcript failed: {e}")

        # fallback: if the session exposes a general input method
        if hasattr(self.session, "on_user_input"):
            try:
                maybe_coro = self.session.on_user_input(user_id, transcript)
                if asyncio.iscoroutine(maybe_coro):
                    self._loop.create_task(maybe_coro)
                return
            except Exception as e:
                print(f"[InterruptHandler] session.on_user_input failed: {e}")

        # If there is no known method, log and drop
        print("[InterruptHandler] Warning: session has no transcript handler. Dropping transcript.")


# ------------------------- Integration snippet -------------------------

def integration_snippet(session):
    """Example wiring — edit to your session's event names.

    Example (pseudo):

        ih = InterruptHandler(session)
        session.register_vad_callback(ih.on_vad_start)
        session.register_stt_partial_callback(ih.on_stt_partial)
        session.register_stt_final_callback(ih.on_stt_final)

    The exact method names differ across STT/VAD integrations. The LiveKit Agents
    repo has a `session` that accepts `vad=...` in the constructor; look for the
    place where VAD and STT events are converted into session events and attach
    the callbacks there.
    """
    ih = InterruptHandler(session)
    # PSEUDO-CALLS: replace with your session's real methods
    if hasattr(session, "register_vad_callback"):
        session.register_vad_callback(ih.on_vad_start)
    if hasattr(session, "register_stt_partial_callback"):
        session.register_stt_partial_callback(ih.on_stt_partial)
    if hasattr(session, "register_stt_final_callback"):
        session.register_stt_final_callback(ih.on_stt_final)
    return ih



