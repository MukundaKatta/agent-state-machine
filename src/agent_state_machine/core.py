"""Generic finite state machine for agent workflow states.

:class:`StateMachine` tracks which state an agent (or workflow) is in and
enforces that transitions happen only along declared edges.  Each transition
is triggered by an *event* string.  The machine records a full
:attr:`~StateMachine.history` of ``(from_state, event, to_state)`` tuples.

Example::

    sm = StateMachine()
    sm.add_state("idle",       is_initial=True)
    sm.add_state("thinking")
    sm.add_state("responding")
    sm.add_state("done",       is_terminal=True)

    sm.add_transition("idle",       "thinking",   "start")
    sm.add_transition("thinking",   "responding", "reply")
    sm.add_transition("responding", "done",       "finish")
    sm.add_transition("thinking",   "idle",       "reset")

    sm.trigger("start")    # idle → thinking
    sm.trigger("reply")    # thinking → responding
    sm.trigger("finish")   # responding → done
    assert sm.is_terminal()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class StateMachineError(RuntimeError):
    """Base for state-machine errors."""


class InitialStateNotSetError(StateMachineError):
    """Raised when trigger/reset is called before an initial state is set."""


class StateNotFoundError(StateMachineError, KeyError):
    """Raised when a referenced state name is not registered."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"State {name!r} is not registered.")


class InvalidEventError(StateMachineError):
    """Raised when the event is not allowed from the current state."""

    def __init__(self, event: str, current_state: str) -> None:
        self.event = event
        self.current_state = current_state
        super().__init__(f"Event {event!r} is not valid from state {current_state!r}.")


class DuplicateTransitionError(StateMachineError):
    """Raised when a (from_state, event) pair is already registered."""

    def __init__(self, from_state: str, event: str) -> None:
        self.from_state = from_state
        self.event = event
        super().__init__(
            f"Transition ({from_state!r}, {event!r}) is already registered."
        )


class SelfLoopError(StateMachineError):
    """Raised when a self-loop transition is declared but not allowed."""

    def __init__(self, state: str, event: str) -> None:
        self.state = state
        self.event = event
        super().__init__(
            f"Self-loop transition ({state!r}, {event!r}) is not allowed; "
            "pass allow_self_loops=True to permit it."
        )


@dataclass
class _StateInfo:
    name: str
    is_terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "is_terminal": self.is_terminal,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _StateInfo:
        return cls(
            name=data["name"],
            is_terminal=bool(data.get("is_terminal", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class _HistoryEntry:
    from_state: str
    event: str
    to_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_state": self.from_state,
            "event": self.event,
            "to_state": self.to_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _HistoryEntry:
        return cls(
            from_state=data["from_state"],
            event=data["event"],
            to_state=data["to_state"],
        )


class StateMachine:
    """A finite state machine for agent workflow state tracking.

    States are registered with :meth:`add_state`.  Transitions are declared
    with :meth:`add_transition`.  Events are fired with :meth:`trigger`.

    The machine maintains a :attr:`history` of all transitions taken.

    Args:
        allow_self_loops: If ``True``, a state can transition to itself.
            Default ``False`` (self-loop transitions are rejected at
            :meth:`add_transition` time).

    Example::

        sm = StateMachine()
        sm.add_state("idle",  is_initial=True)
        sm.add_state("busy")
        sm.add_state("done",  is_terminal=True)
        sm.add_transition("idle", "busy", "start")
        sm.add_transition("busy", "done", "finish")
        sm.trigger("start")
        sm.trigger("finish")
        assert sm.is_terminal()
    """

    def __init__(self, *, allow_self_loops: bool = False) -> None:
        self._states: dict[str, _StateInfo] = {}
        # (from_state, event) → to_state
        self._transitions: dict[tuple[str, str], str] = {}
        self._initial: str | None = None
        self._current: str | None = None
        self._history: list[_HistoryEntry] = []
        self._allow_self_loops = allow_self_loops

    # ------------------------------------------------------------------
    # Building the machine
    # ------------------------------------------------------------------

    def add_state(
        self,
        name: str,
        *,
        is_initial: bool = False,
        is_terminal: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> StateMachine:
        """Register a state.

        Args:
            name:        Unique state name.
            is_initial:  Mark this as the starting state (only one allowed).
            is_terminal: Marks this as a terminal/accepting state.
            metadata:    Arbitrary extra data attached to the state.

        Returns:
            ``self`` for chaining.
        """
        self._states[name] = _StateInfo(
            name=name,
            is_terminal=is_terminal,
            metadata=dict(metadata or {}),
        )
        if is_initial:
            self._initial = name
            if self._current is None:
                self._current = name
        return self

    def set_initial(self, name: str) -> StateMachine:
        """Set the initial state (without re-registering it).

        Raises:
            StateNotFoundError: If *name* is not registered.
        """
        if name not in self._states:
            raise StateNotFoundError(name)
        self._initial = name
        if self._current is None:
            self._current = name
        return self

    def add_transition(
        self,
        from_state: str,
        to_state: str,
        event: str,
    ) -> StateMachine:
        """Declare a transition.

        Args:
            from_state: Origin state name.
            to_state:   Destination state name.
            event:      Event string that fires this transition.

        Returns:
            ``self`` for chaining.

        Raises:
            StateNotFoundError: If either state is not registered.
            SelfLoopError: If *from_state* equals *to_state* and
                ``allow_self_loops`` is ``False``.
            DuplicateTransitionError: If this (from_state, event) pair is
                already registered.
        """
        if from_state not in self._states:
            raise StateNotFoundError(from_state)
        if to_state not in self._states:
            raise StateNotFoundError(to_state)
        if from_state == to_state and not self._allow_self_loops:
            raise SelfLoopError(from_state, event)
        key = (from_state, event)
        if key in self._transitions:
            raise DuplicateTransitionError(from_state, event)
        self._transitions[key] = to_state
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def trigger(self, event: str) -> str:
        """Fire *event* and advance to the next state.

        Args:
            event: The event to fire.

        Returns:
            The name of the new current state.

        Raises:
            InitialStateNotSetError: If no initial state has been set.
            InvalidEventError: If *event* is not valid from the current state.
        """
        if self._current is None:
            raise InitialStateNotSetError(
                "No initial state set. Call add_state(..., is_initial=True)"
                " or set_initial()."
            )
        key = (self._current, event)
        if key not in self._transitions:
            raise InvalidEventError(event, self._current)
        from_state = self._current
        to_state = self._transitions[key]
        self._history.append(
            _HistoryEntry(from_state=from_state, event=event, to_state=to_state)
        )
        self._current = to_state
        return to_state

    def can_trigger(self, event: str) -> bool:
        """Return ``True`` if *event* is valid from the current state."""
        if self._current is None:
            return False
        return (self._current, event) in self._transitions

    def available_events(self) -> list[str]:
        """Return all events that can be triggered from the current state."""
        if self._current is None:
            return []
        return sorted(
            event for (state, event) in self._transitions if state == self._current
        )

    def reset(self) -> str:
        """Return to the initial state and clear history.

        Returns:
            The initial state name.

        Raises:
            InitialStateNotSetError: If no initial state is set.
        """
        if self._initial is None:
            raise InitialStateNotSetError(
                "No initial state set. Call add_state(..., is_initial=True)"
                " or set_initial()."
            )
        self._current = self._initial
        self._history.clear()
        return self._current

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def current(self) -> str | None:
        """Current state name, or ``None`` if not started."""
        return self._current

    def is_terminal(self) -> bool:
        """Return ``True`` if the current state is terminal."""
        if self._current is None:
            return False
        return self._states[self._current].is_terminal

    def state_info(self, name: str) -> _StateInfo:
        """Return the :class:`_StateInfo` for *name*.

        Raises:
            StateNotFoundError: If not found.
        """
        if name not in self._states:
            raise StateNotFoundError(name)
        return self._states[name]

    @property
    def history(self) -> list[_HistoryEntry]:
        """History of transitions as ``(from_state, event, to_state)`` entries."""
        return list(self._history)

    def states(self) -> list[str]:
        """All registered state names in insertion order."""
        return list(self._states)

    def __contains__(self, name: str) -> bool:
        return name in self._states

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the machine to a plain dict."""
        return {
            "allow_self_loops": self._allow_self_loops,
            "initial": self._initial,
            "current": self._current,
            "states": [s.to_dict() for s in self._states.values()],
            "transitions": [
                {"from_state": k[0], "event": k[1], "to_state": v}
                for k, v in self._transitions.items()
            ],
            "history": [e.to_dict() for e in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateMachine:
        """Reconstruct a :class:`StateMachine` from a plain dict."""
        sm = cls(allow_self_loops=bool(data.get("allow_self_loops", False)))
        for sd in data.get("states", []):
            s = _StateInfo.from_dict(sd)
            sm._states[s.name] = s
        sm._initial = data.get("initial")
        sm._current = data.get("current")
        for td in data.get("transitions", []):
            sm._transitions[(td["from_state"], td["event"])] = td["to_state"]
        for hd in data.get("history", []):
            sm._history.append(_HistoryEntry.from_dict(hd))
        return sm

    def __repr__(self) -> str:
        return (
            f"StateMachine(states={len(self._states)},"
            f" current={self._current!r},"
            f" transitions={len(self._transitions)})"
        )
