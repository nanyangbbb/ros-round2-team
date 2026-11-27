"""Regression for the observed five-cone layout that exhausted finish preplanning."""
import math
from pathlib import Path

import rospkg
import yaml

from finish_region import contained
from hybrid_path import FootprintMap, fixed_walls
from hybrid_trial import plan_finish_path


def main():
    packages = rospkg.RosPack()
    config = Path(packages.get_path('round2_baseline')) / 'config'
    footprint = yaml.safe_load((config / 'navigation.yaml').read_text())['local_costmap']['footprint']
    route = {name: (x, y, yaw) for name, x, y, yaw in yaml.safe_load((config / 'route.yaml').read_text())}
    walls = tuple(fixed_walls(Path(packages.get_path('car3')) / 'world' / 'round2.world'))

    # Measured block stop and lidar-confirmed cells from the third official random trial.
    start = (-1.7392809439368322, 2.228909634511499, 1.980293752970798)
    occupied = frozenset({(1, 1), (1, 2), (2, 1), (3, 1), (5, 2)})
    goal = route['finish']
    approach_distance = math.dist(route['finish_approach'][:2], goal[:2])
    result = plan_finish_path(start, goal, approach_distance, walls, footprint, occupied, lambda: False)
    assert result is not None, 'Finish preplanning exhausted its searches from the recorded block stop'
    path, _, planned_obstacles, _, cone_margin, tail = result
    assert planned_obstacles == occupied and cone_margin >= .035
    assert math.isclose(tail, .80) and math.dist(start, path[0]) < 1e-9
    assert math.dist(goal, path[-1]) < 1e-9
    checked = FootprintMap(walls, footprint, occupied, margin=.014, cone_margin=.035)
    assert checked.sweep_clear(path), 'The optimized return path violates a footprint clearance'
    assert contained(*path[-1], footprint), 'The complete vehicle must fit inside the finish box'
    print('PASS: preplanned the observed round 3 return with checked clearance and finish containment')


if __name__ == '__main__':
    main()
