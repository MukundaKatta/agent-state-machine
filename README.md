# agent-state-machine

Generic finite state machine for agent workflow states.

Define states and transitions, then fire events to advance the machine. Tracks history of all transitions. Useful for modeling agent lifecycle, conversation state, or multi-step workflow phases.

## Install

```bash
pip install agent-state-machine
```

## Quick start

```python
from agent_state_machine import StateMachine

sm = StateMachine()
sm.add_state("idle",       is_initial=True)
sm.add_state("thinking")
sm.add_state("responding")
sm.add_state("done",       is_terminal=True)

sm.add_transition("idle",       "thinking",   "start")
sm.add_transition("thinking",   "responding", "reply")
sm.add_transition("responding", "done",       "finish")
sm.add_transition("thinking",   "idle",       "cancel")

sm.trigger("start")     # idle → thinking
sm.trigger("reply")     # thinking → responding
sm.trigger("finish")    # responding → done
assert sm.is_terminal()
```

## API

### `StateMachine`

| Method | Description |
|---|---|
| `add_state(name, *, is_initial, is_terminal, metadata)` | Register a state |
| `set_initial(name)` | Set initial state on an existing state |
| `add_transition(from_state, to_state, event)` | Declare a transition |
| `trigger(event)` | Fire event; returns new state name |
| `can_trigger(event)` | Is event valid from current state? |
| `available_events()` | Events triggerable from current state |
| `current` | Current state name |
| `is_terminal()` | Is current state terminal? |
| `reset()` | Return to initial state, clear history |
| `history` | List of `(from_state, event, to_state)` entries |
| `states()` | All registered state names |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

### Errors

- `StateNotFoundError` — state name not registered
- `InvalidEventError` — event not valid from current state
- `DuplicateTransitionError` — transition already declared
- `SelfLoopError` — self-loop transition declared while `allow_self_loops=False`
- `InitialStateNotSetError` — trigger/reset before initial state set

## License

MIT
