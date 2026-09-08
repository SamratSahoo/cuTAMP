"""Tests for _plan_transit's hidden-obstacle scoping.

Placing INTO a container needs the container hidden for the leg that arrives inside it, and NOT for
the traverse that gets there -- otherwise cuRobo either has no IK for the pre-place pose (IK_FAIL on
every particle) or is free to sweep the held object straight through the container's wall.
"""

import torch

from cutamp.motion_solver import _plan_transit


class _Plan:
    def __init__(self, position):
        self.position = position


class _Result:
    def __init__(self, success, status="OK"):
        self.success = success
        self.status = status
        self.interpolation_dt = 0.02
        self.optimized_plan = None
        self.optimized_dt = 0.02

    def get_interpolated_plan(self):
        return _Plan(torch.zeros(2, 7))


class _Checker:
    """Records which obstacles were disabled at each plan_single call."""

    def __init__(self):
        self.disabled = set()

    def enable_obstacle(self, name, enable):
        self.disabled.discard(name) if enable else self.disabled.add(name)


class _MotionGen:
    def __init__(self, fail_while_visible=()):
        self.world_coll_checker = _Checker()
        self.calls = []  # (goal z, frozenset of obstacles hidden at the time)
        self._fail_while_visible = set(fail_while_visible)

    def plan_single(self, start_js, goal_pose, plan_config):
        hidden = frozenset(self.world_coll_checker.disabled)
        z = float(goal_pose.position[0, 2]) if goal_pose.position.ndim == 2 else float(goal_pose.position[2])
        self.calls.append((round(z, 3), hidden))
        # A goal below the rim is unreachable unless the container is hidden -- the IK_FAIL.
        blocked = z < 0.1 and bool(self._fail_while_visible - hidden)
        return _Result(not blocked, "IK_FAIL" if blocked else "OK")


class _Kin:
    def get_state(self, position):
        class _S:
            class ee_pose:
                @staticmethod
                def get_matrix():
                    start = torch.eye(4)[None].clone()
                    start[0, :3, 3] = torch.tensor([0.0, 0.0, 0.4])
                    return start

        return _S()


class _World:
    kin_model = _Kin()


def _goal(z):
    goal = torch.eye(4)
    goal[:3, 3] = torch.tensor([0.5, 0.0, z])
    return goal


def _start_js():
    from curobo.types.state import JointState

    return JointState.from_position(torch.zeros(1, 7))


def test_the_traverse_keeps_the_container_and_the_arrival_does_not():
    motion_gen = _MotionGen(fail_while_visible={"box"})
    results, last = _plan_transit(
        motion_gen, _start_js(), _goal(0.02), plan_config=None, world=_World(),
        apex_height=0.075, apex_min_dist=0.10, hidden_at_goal=("box",),
    )
    assert results is not None and last.success
    # Two legs: up to the apex with the box visible, then down into it with the box hidden.
    apex_call, descend_call = motion_gen.calls
    assert apex_call[1] == frozenset()  # traverse still routes around the container
    assert descend_call[1] == frozenset({"box"})
    assert apex_call[0] > descend_call[0]  # and it really is an arc over the top


def test_the_container_is_restored_afterwards():
    motion_gen = _MotionGen(fail_while_visible={"box"})
    _plan_transit(
        motion_gen, _start_js(), _goal(0.02), None, _World(), 0.075, 0.10, hidden_at_goal=("box",)
    )
    assert motion_gen.world_coll_checker.disabled == set()


def test_the_direct_fallback_also_hides_it():
    # No apex (the goal is too close horizontally), so the direct plan is the arriving leg.
    motion_gen = _MotionGen(fail_while_visible={"box"})
    results, _ = _plan_transit(
        motion_gen, _start_js(), _goal(0.02), None, _World(), 0.075, apex_min_dist=10.0,
        hidden_at_goal=("box",),
    )
    assert results is not None
    assert motion_gen.calls == [(0.02, frozenset({"box"}))]


def test_nothing_is_hidden_by_default():
    """Placing ON a surface is unchanged: every leg keeps the full world."""
    motion_gen = _MotionGen()
    results, _ = _plan_transit(motion_gen, _start_js(), _goal(0.3), None, _World(), 0.075, 0.10)
    assert results is not None
    assert all(hidden == frozenset() for _, hidden in motion_gen.calls)


def test_an_unreachable_goal_still_fails_when_nothing_is_hidden():
    """The IK_FAIL this exists to fix, reproduced: the same goal without the exemption."""
    motion_gen = _MotionGen(fail_while_visible={"box"})
    results, last = _plan_transit(
        motion_gen, _start_js(), _goal(0.02), None, _World(), 0.075, 0.10, hidden_at_goal=()
    )
    assert results is None
    assert last.status == "IK_FAIL"


class _Lifter:
    """A motion_gen whose plans succeed, and whose configurations are only legal above ``clears``.

    Stands in for the geometry that matters: the gripper ends the plan down inside a container's
    filled hull, and only a long enough lift gets it out.
    """

    def __init__(self, clears=0.18, plan_fails_above=None):
        self.world_coll_checker = _Checker()
        self.clears = clears
        self.plan_fails_above = plan_fails_above
        self.attempts = []  # lift height of each plan_single, and what was hidden
        self.checked = []  # heights passed to check_start_state

    def plan_single(self, start_js, goal_pose, plan_config):
        height = float(goal_pose.position[0, 2])
        self.attempts.append((round(height, 3), frozenset(self.world_coll_checker.disabled)))
        if self.plan_fails_above is not None and height > self.plan_fails_above:
            return _Result(False, "IK_FAIL")
        result = _Result(True)
        result.get_interpolated_plan = lambda h=height: _Plan(torch.full((2, 7), h))
        return result

    def check_start_state(self, js):
        height = float(js.position[0, 0])
        self.checked.append(round(height, 3))
        return (height >= self.clears, None)


def _ladder(base_z=0.0):
    """Four candidate lifts, 5 to 20 cm, as solve_curobo builds them."""
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 2, 3] = torch.tensor([0.05, 0.10, 0.15, 0.20]) + base_z
    return poses


class TestPlanRetractOutOf:
    def test_climbs_until_the_lift_actually_leaves_the_container(self):
        from cutamp.motion_solver import _plan_retract_out_of

        motion_gen = _Lifter(clears=0.18)
        result, _ = _plan_retract_out_of(motion_gen, _start_js(), _ladder(), None, ("box",))
        assert result is not None
        # It kept going past the lifts that planned but stayed inside the hull.
        assert [h for h, _ in motion_gen.attempts] == [0.05, 0.10, 0.15, 0.20]
        assert motion_gen.checked == [0.05, 0.10, 0.15, 0.20]

    def test_stops_at_the_first_lift_that_clears(self):
        from cutamp.motion_solver import _plan_retract_out_of

        motion_gen = _Lifter(clears=0.08)
        result, _ = _plan_retract_out_of(motion_gen, _start_js(), _ladder(), None, ("box",))
        assert result is not None
        assert [h for h, _ in motion_gen.attempts] == [0.05, 0.10]

    def test_the_container_is_hidden_for_the_lift_and_restored_for_the_check(self):
        from cutamp.motion_solver import _plan_retract_out_of

        motion_gen = _Lifter(clears=0.08)
        _plan_retract_out_of(motion_gen, _start_js(), _ladder(), None, ("box",))
        # Hidden while planning -- the start state is inside the hull and cuRobo would refuse it.
        assert all(hidden == frozenset({"box"}) for _, hidden in motion_gen.attempts)
        # And back on afterwards, which is what makes check_start_state mean anything.
        assert motion_gen.world_coll_checker.disabled == set()

    def test_falls_back_to_the_longest_lift_that_planned(self):
        from cutamp.motion_solver import _plan_retract_out_of

        # Nothing clears, and the two longest lifts do not even plan.
        motion_gen = _Lifter(clears=10.0, plan_fails_above=0.10)
        result, status = _plan_retract_out_of(motion_gen, _start_js(), _ladder(), None, ("box",))
        assert result is not None and result.success  # the 10 cm one
        assert status == "IK_FAIL"  # from the last attempt, for the error message if it mattered

    def test_reports_failure_when_no_lift_plans_at_all(self):
        from cutamp.motion_solver import _plan_retract_out_of

        motion_gen = _Lifter(clears=10.0, plan_fails_above=0.0)
        result, status = _plan_retract_out_of(motion_gen, _start_js(), _ladder(), None, ("box",))
        assert result is None
        assert status == "IK_FAIL"
