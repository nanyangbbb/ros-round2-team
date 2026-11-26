"""A route that finishes planning as the car stops must survive the viewpoint handoff."""
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import hybrid_trial as trial
from hybrid_path import CANDIDATES, NoPath, legal_layouts


def check_handoff(abort=False):
    layouts = tuple(legal_layouts())[:2]
    observations = {}
    table = SimpleNamespace(snapshot=lambda: dict(observations))
    footprint = [(-.12, -.08), (.12, -.08), (.12, .08), (-.12, .08)]
    pose = (.10, 1.5, 0.)
    viewpoint = (.50, 1.5, 0.)
    goal = (1., 1.5, 0.)
    reached = threading.Event()
    searching = threading.Event()
    cancelled_search = threading.Event()
    searches = []
    driven = []

    def search(start, target, course, cancelled, *, viewpoint=None, **kwargs):
        if viewpoint:
            return trial.rounded_path([start[:2], (.50, 1.5)], spacing=.02), 1
        if len(course.obstacles) > 5:
            raise NoPath('No common route yet')
        searches.append(start)
        searching.set()
        if abort:
            until = time.monotonic() + 2
            while not cancelled() and time.monotonic() < until:
                time.sleep(.005)
            if cancelled():
                cancelled_search.set()
            raise NoPath('Stopped search')
        assert reached.wait(2), 'The background search must overlap the observation drive'
        return trial.rounded_path([start[:2], target[:2]], spacing=.02), 1

    def follow(route, deadline, prepared, collision_map, **kwargs):
        nonlocal pose
        assert collision_map.sweep_clear(prepared.path)
        driven.append(route[0][0])
        if driven[-1] == 'cone_viewpoint':
            pose = viewpoint
            observations.update({cell: cell in layouts[0] for cell in CANDIDATES})
            assert kwargs['select_path']() is None
            if abort:
                assert searching.wait(2)
                raise RuntimeError('Injected control failure')
            reached.set()
        else:
            assert tuple(prepared.path[0]) == viewpoint and tuple(prepared.path[-1]) == goal

    odom = SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=0.))))
    control = SimpleNamespace(vision_enabled=False, odom=odom, follow_bends=follow, report=lambda *_: None)
    with patch.object(trial, 'actual_pose', lambda _: pose), patch.object(trial, 'hybrid_astar', search), \
            patch.object(trial.rospy, 'get_time', time.monotonic):
        try:
            trial.drive_blocks(control, None, lambda: False, table, (), footprint, layouts, (), goal, {},
                               local_geometry=True, optimize=True)
        except RuntimeError as exc:
            assert abort and str(exc) == 'Injected control failure'
            assert cancelled_search.is_set() and driven == ['cone_viewpoint']
            return
    assert not abort
    assert driven == ['cone_viewpoint', 'blocks'], driven
    assert len(searches) == 1, f'Repeated the resolved layout search after stopping: {searches}'


if __name__ == '__main__':
    check_handoff()
    check_handoff(abort=True)
    print('PASS: reused the checked five-cone route at the viewpoint and cancelled work on abort')
