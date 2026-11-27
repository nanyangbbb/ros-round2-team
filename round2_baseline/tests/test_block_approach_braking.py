"""Replay the moving handoff and braking corridor from the fifth official random trial."""
import math
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import rospkg
import yaml

import hybrid_trial as trial
from bend_path import PreparedPath
from hybrid_path import CANDIDATES, FootprintMap, NoPath, fixed_walls, legal_layouts, transform


def main():
    packages = rospkg.RosPack()
    config = Path(packages.get_path('round2_baseline')) / 'config'
    walls = tuple(fixed_walls(Path(packages.get_path('car3')) / 'world/round2.world'))
    footprint = yaml.safe_load((config / 'navigation.yaml').read_text())['local_costmap']['footprint']
    occupied = frozenset({(1, 1), (1, 2), (1, 4), (2, 1), (2, 4)})
    alternate = next(layout for layout in legal_layouts() if layout != occupied)
    observations = {}
    table = SimpleNamespace(snapshot=lambda: dict(observations))
    start = (-.38418677308935206, .5926872620002067, -2.9318993455584565)
    observation_end = (-.4090875825137837, .5839165203774536, 3.4185685903019336)
    join_speed = .06629385271026454
    odom = SimpleNamespace(twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=join_speed))))
    goal = (-1.75, 2.25, 1.57)
    search = trial.hybrid_astar
    planning_map = FootprintMap(walls, footprint, occupied, margin=.040, cone_margin=.050)
    braking_map = FootprintMap(walls, footprint, occupied, margin=.015, cone_margin=.015)

    def require_observation(pose, target, obstacle_map, cancelled, *, viewpoint=None, **kwargs):
        if viewpoint is not None:
            return [start, observation_end], 1
        if len(obstacle_map.obstacles) > len(walls) + len(occupied):
            raise NoPath('Fixture requires observing the fifth cone')
        return search(pose, target, obstacle_map, cancelled, **kwargs)

    def verify(reference, hard_map):
        assert math.dist(reference.path[-1][:2], goal[:2]) < 1e-9
        assert reference.parking_tolerance == .025
        assert all(margin == 0. for _, _, _, margin in hard_map.obstacles)
        assert planning_map.sweep_clear(reference.path), 'Insufficient planned cone or wall clearance'
        uncapped = PreparedPath(reference.path, *trial.HYBRID_LIMITS, start_speed=reference.speeds[0],
                                lookahead_time=reference.lookahead_time, parking_tolerance=reference.parking_tolerance)
        caps = []
        for i, pose in enumerate(reference.path):
            if not 1.24 < pose[1] < 1.53:
                continue
            # The guard drops yaw as translation slews to zero. Check that entire tangent.
            speed = reference.speeds[i]
            distance = .10 * speed + speed ** 2 / (2 * trial.HYBRID_LIMITS[1])
            stop_pose = (*transform(*pose, distance, 0.), pose[2])
            assert braking_map.sweep_clear([pose, stop_pose]), ('Insufficient braking clearance', i)
            normal_speed, _, _ = uncapped.command(i, *pose, .5)
            issued_speed, _, _ = reference.command(i, *pose, .5)
            caps.append(normal_speed - issued_speed)
        assert caps and max(caps) > .05, 'The speed envelope must slow before the close cone'

    def follow(route, deadline, prepared, collision_map, **kwargs):
        if route[0][0] == 'blocks':
            assert kwargs['local_velocity']
            verify(prepared, collision_map)
            return
        assert kwargs['local_velocity']
        observations.update({cell: cell in occupied for cell in CANDIDATES})
        until = time.monotonic() + 15
        while time.monotonic() < until:
            selected = kwargs['select_path']()
            if selected is not None:
                name, _, _, _, reference, hard_map, _ = selected
                assert name == 'blocks' and tuple(reference.path[0]) == start
                verify(reference, hard_map)
                return
            time.sleep(.005)
        raise AssertionError('Missing moving handoff')

    control = SimpleNamespace(vision_enabled=False, odom=odom, follow_bends=follow, report=lambda *_: None)
    with patch.object(trial, 'actual_pose', lambda _: start), patch.object(trial.rospy, 'get_time', time.monotonic), \
            patch.object(trial, 'hybrid_astar', require_observation):
        for known in ((), occupied):
            evidence = {}
            trial.drive_blocks(control, None, lambda: False, table, walls, footprint, (occupied, alternate),
                               known, goal, evidence, local_geometry=True, optimize=True)
            assert bool(evidence['exploration']) == (not known)
    print('PASS: moving and direct block routes retain clearance, slow before the cone and keep the parking bound')


if __name__ == '__main__':
    main()
