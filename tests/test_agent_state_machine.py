"""Tests for agent-state-machine."""

from __future__ import annotations

import pytest

from agent_state_machine import StateMachine
from agent_state_machine.core import (
    DuplicateTransitionError,
    InitialStateNotSetError,
    InvalidEventError,
    StateNotFoundError,
)


def _simple_sm() -> StateMachine:
    """idle → thinking → done."""
    sm = StateMachine()
    sm.add_state("idle", is_initial=True)
    sm.add_state("thinking")
    sm.add_state("done", is_terminal=True)
    sm.add_transition("idle", "thinking", "start")
    sm.add_transition("thinking", "done", "finish")
    sm.add_transition("thinking", "idle", "reset")
    return sm


# ---------------------------------------------------------------------------
# add_state
# ---------------------------------------------------------------------------


def test_add_state_registers():
    sm = StateMachine()
    sm.add_state("idle")
    assert "idle" in sm


def test_add_initial_sets_current():
    sm = StateMachine()
    sm.add_state("idle", is_initial=True)
    assert sm.current == "idle"


def test_add_state_chaining():
    sm = StateMachine()
    result = sm.add_state("idle", is_initial=True)
    assert result is sm


def test_add_terminal_state():
    sm = StateMachine()
    sm.add_state("done", is_terminal=True)
    assert sm.state_info("done").is_terminal


def test_add_state_with_metadata():
    sm = StateMachine()
    sm.add_state("idle", metadata={"label": "Waiting"})
    assert sm.state_info("idle").metadata == {"label": "Waiting"}


# ---------------------------------------------------------------------------
# set_initial
# ---------------------------------------------------------------------------


def test_set_initial():
    sm = StateMachine()
    sm.add_state("idle")
    sm.set_initial("idle")
    assert sm.current == "idle"


def test_set_initial_missing_raises():
    sm = StateMachine()
    with pytest.raises(StateNotFoundError) as exc_info:
        sm.set_initial("nope")
    assert exc_info.value.name == "nope"


# ---------------------------------------------------------------------------
# add_transition
# ---------------------------------------------------------------------------


def test_add_transition():
    sm = StateMachine()
    sm.add_state("a", is_initial=True)
    sm.add_state("b")
    sm.add_transition("a", "b", "go")
    assert sm.can_trigger("go")


def test_add_transition_chaining():
    sm = StateMachine()
    sm.add_state("a", is_initial=True)
    sm.add_state("b")
    result = sm.add_transition("a", "b", "go")
    assert result is sm


def test_add_transition_missing_from_state_raises():
    sm = StateMachine()
    sm.add_state("b")
    with pytest.raises(StateNotFoundError):
        sm.add_transition("missing", "b", "go")


def test_add_transition_missing_to_state_raises():
    sm = StateMachine()
    sm.add_state("a", is_initial=True)
    with pytest.raises(StateNotFoundError):
        sm.add_transition("a", "missing", "go")


def test_add_duplicate_transition_raises():
    sm = StateMachine()
    sm.add_state("a", is_initial=True)
    sm.add_state("b")
    sm.add_transition("a", "b", "go")
    with pytest.raises(DuplicateTransitionError) as exc_info:
        sm.add_transition("a", "b", "go")
    assert exc_info.value.from_state == "a"
    assert exc_info.value.event == "go"


# ---------------------------------------------------------------------------
# trigger
# ---------------------------------------------------------------------------


def test_trigger_advances_state():
    sm = _simple_sm()
    new_state = sm.trigger("start")
    assert new_state == "thinking"
    assert sm.current == "thinking"


def test_trigger_records_history():
    sm = _simple_sm()
    sm.trigger("start")
    h = sm.history
    assert len(h) == 1
    assert h[0].from_state == "idle"
    assert h[0].event == "start"
    assert h[0].to_state == "thinking"


def test_trigger_invalid_event_raises():
    sm = _simple_sm()
    with pytest.raises(InvalidEventError) as exc_info:
        sm.trigger("finish")  # can't finish from idle
    assert exc_info.value.event == "finish"
    assert exc_info.value.current_state == "idle"


def test_trigger_no_initial_state_raises():
    sm = StateMachine()
    sm.add_state("a")
    sm.add_state("b")
    sm.add_transition("a", "b", "go")
    with pytest.raises(InitialStateNotSetError):
        sm.trigger("go")


def test_trigger_multiple_transitions():
    sm = _simple_sm()
    sm.trigger("start")
    sm.trigger("finish")
    assert sm.current == "done"
    assert len(sm.history) == 2


def test_trigger_cycle():
    sm = _simple_sm()
    sm.trigger("start")
    sm.trigger("reset")
    assert sm.current == "idle"


# ---------------------------------------------------------------------------
# can_trigger / available_events
# ---------------------------------------------------------------------------


def test_can_trigger_true():
    sm = _simple_sm()
    assert sm.can_trigger("start")


def test_can_trigger_false():
    sm = _simple_sm()
    assert not sm.can_trigger("finish")


def test_can_trigger_no_state():
    sm = StateMachine()
    sm.add_state("a")
    # current is None since no is_initial and set_initial not called
    sm2 = StateMachine()
    assert not sm2.can_trigger("any")


def test_available_events():
    sm = _simple_sm()
    sm.trigger("start")
    events = sm.available_events()
    assert set(events) == {"finish", "reset"}


def test_available_events_empty_no_state():
    sm = StateMachine()
    assert sm.available_events() == []


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


def test_is_terminal_false():
    sm = _simple_sm()
    assert not sm.is_terminal()


def test_is_terminal_true():
    sm = _simple_sm()
    sm.trigger("start")
    sm.trigger("finish")
    assert sm.is_terminal()


def test_is_terminal_no_state():
    sm = StateMachine()
    assert not sm.is_terminal()


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_returns_to_initial():
    sm = _simple_sm()
    sm.trigger("start")
    sm.reset()
    assert sm.current == "idle"


def test_reset_clears_history():
    sm = _simple_sm()
    sm.trigger("start")
    sm.reset()
    assert sm.history == []


def test_reset_no_initial_raises():
    sm = StateMachine()
    sm.add_state("a")
    with pytest.raises(InitialStateNotSetError):
        sm.reset()


# ---------------------------------------------------------------------------
# states / state_info
# ---------------------------------------------------------------------------


def test_states_list():
    sm = _simple_sm()
    assert sm.states() == ["idle", "thinking", "done"]


def test_state_info_not_found_raises():
    sm = StateMachine()
    with pytest.raises(StateNotFoundError):
        sm.state_info("nope")


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_to_dict_round_trip():
    sm = _simple_sm()
    sm.trigger("start")

    restored = StateMachine.from_dict(sm.to_dict())
    assert restored.current == "thinking"
    assert len(restored.states()) == 3
    assert restored.can_trigger("finish")
    assert not restored.can_trigger("start")
    assert len(restored.history) == 1


def test_from_dict_preserves_terminal():
    sm = _simple_sm()
    sm.trigger("start")
    sm.trigger("finish")
    restored = StateMachine.from_dict(sm.to_dict())
    assert restored.is_terminal()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr():
    sm = _simple_sm()
    r = repr(sm)
    assert "StateMachine" in r
    assert "idle" in r
