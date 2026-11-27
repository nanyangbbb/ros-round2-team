"""Regression for the observed five-cone layout that exhausted finish preplanning."""
import math
from pathlib import Path

import rospkg
import yaml

from finish_region import contained
from bend_path import PreparedPath
from hybrid_path import FootprintMap, connector, fixed_walls
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
    check_forward_tracking(path)
    print('PASS: preplanned the observed round 3 return with checked clearance and finish containment')

    # A collinear cubic can reverse between samples even when its sampled curvature is zero.
    assert not list(connector((0., 0., 0.), (.10, 0., math.pi), 5.8))
    start = (-1.7471518851723196, 2.227902064483927, 1.6481464388886182)
    occupied = frozenset({(1, 1), (1, 4), (3, 3), (3, 4), (4, 2)})
    result = plan_finish_path(start, goal, approach_distance, walls, footprint, occupied, lambda: False)
    assert result is not None, 'Round 4 needs a feasible departure after rejecting the cusp'
    path, _, _, _, cone_margin, tail = result
    assert math.dist(start, path[0]) < 1e-9 and math.dist(goal, path[-1]) < 1e-9
    assert math.isclose(tail, .80) and cone_margin >= .035
    assert FootprintMap(walls, footprint, occupied, cone_margin=cone_margin).sweep_clear(path)
    reference = check_forward_tracking(path)
    # Recorded odometry immediately before the old tracker selected a point behind the vehicle.
    speed, _, _ = reference.command(0, -1.74718586, 2.23547576, 1.64846407, .09822)
    assert speed > 0
    print('PASS: round 4 return can be followed forward throughout its checked path')


def check_forward_tracking(path):
    reference = PreparedPath(path, .9, 1.25, 2.2, terminal_heading_distance=.20)
    progress = 0
    for pose in path[:-1]:
        _, _, progress = reference.command(progress, *pose, 0.)
    return reference


if __name__ == '__main__':
    main()
