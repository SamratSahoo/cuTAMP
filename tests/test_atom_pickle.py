"""An unpickled Atom must be usable in this process's sets, not just compare equal.

``Atom.__hash__`` returns ``hash(str(atom))``, and CPython salts string hashing per process. Without
a rehash on unpickle, an Atom restored from a pickle written by another process compares equal to
its freshly built twin while hashing differently -- so it is a SEPARATE member of the same set and
``atom in state`` is False. Everything downstream of ``perception/cutamp_env.pkl`` depends on this:
BFS grounds its goal atoms fresh, so an unpickled ``goal_state`` is unreachable and planning reports
"No valid plan skeletons found for the given goal" on a goal that plans fine live.
"""

import base64
import pickle
import subprocess
import sys

from cutamp.task_planning.base_structs import Atom, Fluent, Parameter

ON = Fluent("On", [Parameter("o", "movable"), Parameter("s", "surface")])

# Pickle one atom under an explicit hash seed and hand back the bytes.
_CHILD = """
import base64, pickle, sys
from cutamp.task_planning.base_structs import Fluent, Parameter
on = Fluent("On", [Parameter("o", "movable"), Parameter("s", "surface")])
sys.stdout.write(base64.b64encode(pickle.dumps(on.ground("banana", "wooden_tray"))).decode())
"""


def test_rehashes_after_a_stale_cached_hash():
    """The mechanism, independent of what seed the test process happens to run under."""
    atom = ON.ground("banana", "wooden_tray")
    stale = pickle.loads(pickle.dumps(atom))
    object.__setattr__(stale, "_cached_hash", 12345)  # what another hash seed looks like

    restored = pickle.loads(pickle.dumps(stale))
    assert restored == atom
    assert hash(restored) == hash(atom)
    assert restored in {atom}
    assert len({atom, restored}) == 1


def test_survives_a_pickle_written_under_a_different_hash_seed():
    """The real case: the pickle was written by the run, and is read back by another process."""
    atom = ON.ground("banana", "wooden_tray")
    goal_state = frozenset({atom})

    # Two seeds, so at least one differs from whatever this process is running under.
    for seed in ("1", "2"):
        out = subprocess.run(
            [sys.executable, "-c", _CHILD],
            capture_output=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        )
        restored = pickle.loads(base64.b64decode(out.stdout))
        assert restored == atom
        assert restored in goal_state, f"unpickled atom not found in the goal state (seed {seed})"
