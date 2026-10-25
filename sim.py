"""ALOHA hand-over rehearsal with measured MuJoCo outcomes.

The controller uses mink differential IK and position actuators. Grasp assistance
is a conditional weld, enabled only near the object and disclosed in metrics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import config
import mink
import mujoco
import numpy as np
from PIL import Image

from schemas import (AttemptMetrics, GeomDesc, HandoverPlan, Keyframe, RolloutBody,
                     RolloutLog, SceneSpec, SimAttempt, SimError)


def _numbers(values):
    return ' '.join(str(float(x)) for x in values)


def _scene_xml(scene: SceneSpec) -> str:
    from robot_descriptions import aloha_mj_description as description
    package = Path(description.PACKAGE_PATH)
    root = ET.parse(package / 'aloha.xml').getroot()
    root.find('compiler').set('meshdir', str(package / 'assets'))
    for include in list(root.findall('include')):
        included = ET.parse(package / include.attrib['file']).getroot()
        root.remove(include)
        for child in included:
            if child.tag != 'keyframe':
                root.append(child)
    option = root.find('option')
    option.set('timestep', '.002')
    option.set('gravity', '0 0 -9.81')
    world = root.find('worldbody')
    for camera in list(world.findall('camera')):
        world.remove(camera)
    for side, pos in [('left', scene.left_arm_base_m), ('right', scene.right_arm_base_m)]:
        world.find(f"body[@name='{side}/base_link']").set('pos', _numbers(pos))
    for body in root.findall('.//body'):
        body.set('gravcomp', '1')
    table_size = [scene.table_size_m[0] / 2, .025] if scene.table_shape == 'round' else [*np.asarray(scene.table_size_m)/2, .025]
    ET.SubElement(world, 'geom', name='desk', type='cylinder' if scene.table_shape == 'round' else 'box',
                  size=_numbers(table_size), pos='0 0 -.025', rgba='.28 .36 .3 1', friction='1 .01 .001')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='2 2 .1',
                  pos=f'0 0 {-scene.table_height_m}', rgba='.1 .13 .12 1')
    ET.SubElement(world, 'camera', name='overhead', pos='0 0 1.65', xyaxes='1 0 0 0 1 0')
    ET.SubElement(world, 'camera', name='front', pos='0 -1.35 .85', xyaxes='1 0 0 0 .55 .84')
    # A virtual wrist-mounted inspection view needs enough clearance to show both
    # fingertips and the object; the previous lens filled the frame with the can.
    eye = np.array([-.08, -.16, .12])
    forward = np.array([.13, 0, 0]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    for side in ['left', 'right']:
        grip = root.find(f".//body[@name='{side}/gripper_link']")
        ET.SubElement(grip, 'camera', name=f'{side}_wrist', pos=_numbers(eye),
                      xyaxes=_numbers(np.r_[right, up]), fovy='65')
    for obj in scene.objects:
        height = obj.size_m[-1]
        body = ET.SubElement(world, 'body', name=obj.name,
                             pos=_numbers([*obj.pos_m[:2], obj.pos_m[2] + height / 2 + .001]),
                             euler=f'0 0 {math.radians(obj.yaw_deg)}')
        if obj.graspable:
            ET.SubElement(body, 'freejoint', name=f'{obj.name}_free')
        size = [obj.size_m[0], height / 2] if obj.kind == 'cylinder' else np.asarray(obj.size_m)/2
        ET.SubElement(body, 'geom', name=f'{obj.name}_geom', type=obj.kind, size=_numbers(size),
                      mass=str(obj.mass_kg), rgba='.87 .53 .27 1' if obj.graspable else '.2 .23 .24 1',
                      friction='1 .01 .001')
    contact = root.find('contact')
    # The disclosed weld replaces fingertip contact only; table and arm contacts remain.
    for side in ['left', 'right']:
        for part in ['gripper_link', 'gripper_base', 'left_finger_link', 'right_finger_link']:
            ET.SubElement(contact, 'exclude', body1=f'{side}/{part}', body2=scene.task_object)
    equality = root.find('equality')
    for side in ['left', 'right']:
        ET.SubElement(equality, 'weld', name=f'{side}_grasp', body1=f'{side}/gripper_link',
                      body2=scene.task_object, active='false', solref='.005 1')
    return ET.tostring(root, encoding='unicode')


def export_mesh_assets(scene: SceneSpec, out_dir: Path) -> dict[int, str]:
    """Export compiled, meter-scale ALOHA meshes for the browser's STL loader."""
    import struct
    model = mujoco.MjModel.from_xml_string(_scene_xml(scene))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for mid in range(model.nmesh):
        start = model.mesh_vertadr[mid]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mid]]
        start = model.mesh_faceadr[mid]
        faces = model.mesh_face[start:start + model.mesh_facenum[mid]]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        records = np.zeros(len(faces), dtype=[('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
        records['normal'], records['vertices'] = normals, triangles
        name = f'aloha-mesh-{mid}.stl'
        with (out_dir / name).open('wb') as handle:
            handle.write(b'ArgonTwin compiled MuJoCo mesh in meters'.ljust(80, b' '))
            handle.write(struct.pack('<I', len(faces)))
            handle.write(records.tobytes())
        result[mid] = name
    (out_dir / 'manifest.json').write_text(json.dumps(result, indent=2))
    return result


def validate_scene(scene: SceneSpec) -> list[str]:
    problems = []
    if any(value <= 0 for value in scene.table_size_m):
        problems.append('Table dimensions must be positive')
    try:
        mujoco.MjModel.from_xml_string(_scene_xml(scene))
    except Exception as exc:
        raise SimError(f'Could not compile the ALOHA scene: {exc}') from exc
    for name, point, base in [('handover_left', scene.handover_point_m, scene.left_arm_base_m),
                              ('handover_right', scene.handover_point_m, scene.right_arm_base_m),
                              ('target', scene.target_m, scene.right_arm_base_m)]:
        if np.linalg.norm(np.asarray(point) - base) > .85:
            problems.append(f'{name} is outside the conservative 0.85 m reach envelope')
    for obj in scene.objects:
        if obj.pos_m[2] < -.01:
            problems.append(f'{obj.name} begins below the tabletop')
    return problems


class _Attempt:
    def __init__(self, scene, plan, out_dir, render):
        self.scene, self.plan, self.out = scene, plan, out_dir
        xml = _scene_xml(scene)
        (out_dir / 'scene.xml').write_text(xml)
        self.model = m = mujoco.MjModel.from_xml_string(xml)
        self.data = d = mujoco.MjData(m)
        neutral = [0, -.96, 1.16, 0, -.3, 0, .035, .035]
        d.qpos[:16] = neutral * 2
        mujoco.mj_forward(m, d)
        self.config = mink.Configuration(m, d.qpos.copy())
        self.tasks = {side: mink.FrameTask(f'{side}/gripper', 'site', 1.0, .01,
                                          lm_damping=1e-4) for side in ['left', 'right']}
        self.posture = mink.PostureTask(m, cost=.002)
        self.posture.set_target(d.qpos.copy())
        self.limits = [mink.ConfigurationLimit(m)]
        self.sites = {side: m.site(f'{side}/gripper').id for side in self.tasks}
        self.grips = {side: m.body(f'{side}/gripper_link').id for side in self.tasks}
        self.eqs = {side: m.equality(f'{side}_grasp').id for side in self.tasks}
        self.obj = m.body(scene.task_object).id
        self.object_spec = next(obj for obj in scene.objects if obj.name == scene.task_object)
        self.robot_geoms = [i for i in range(m.ngeom)
                            if m.body(m.geom_bodyid[i]).name.startswith(('left/', 'right/'))
                            and (m.geom_contype[i] or m.geom_conaffinity[i])]
        self.obstacle_geoms = [m.geom(f'{obj.name}_geom').id for obj in scene.objects if not obj.graspable]
        self.task_geom = m.geom(f'{scene.task_object}_geom').id
        if self.obstacle_geoms:
            self.limits.append(mink.CollisionAvoidanceLimit(
                m, [(self.robot_geoms, self.obstacle_geoms)],
                minimum_distance_from_collisions=.02, collision_detection_distance=.10))
        self.max_obstacle_contact = 0.0
        self.max_arm_penetration = 0.0
        self.max_object_penetration = 0.0
        self.min_obstacle_distance = .10
        self.collision_events = []
        self.closed = {'left': False, 'right': False}
        self.hold_rotations = {}
        self.holds = {'left': False, 'right': False}
        self.grasped = {'left': False, 'right': False}
        self.transferred = False
        self.min_z, self.max_contact = float('inf'), 0.0
        self.failure = None
        self.frames, self.keyframes = [], []
        self.warnings = ['Grasp assist: conditional MuJoCo weld replaces fingertip/object contact; this is not a friction-only grasp.', 'JSON replay approximates robot meshes with bounding boxes; MuJoCo keyframes use the original ALOHA meshes.']
        self.renderer = mujoco.Renderer(m, height=360, width=640) if render else None
        self.body_ids = [i for i in range(m.nbody) if i == 0 or m.body(i).name]
        self.bodies = self._body_descriptions()
        if all((Path(__file__).resolve().parent / 'web/assets/meshes' / f'aloha-mesh-{mid}.stl').is_file() for mid in range(m.nmesh)):
            self.warnings = self.warnings[:1]
        self.act_qpos = [m.jnt_qposadr[j] for j in m.actuator_trnid[:, 0]]
        for side in self.tasks:
            self.tasks[side].set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(d.site_xmat[self.sites[side]].reshape(3, 3)), self.site(side)))

    def site(self, side):
        return self.data.site_xpos[self.sites[side]].copy()

    def _body_descriptions(self):
        result = []
        types = {int(mujoco.mjtGeom.mjGEOM_BOX): 'box', int(mujoco.mjtGeom.mjGEOM_SPHERE): 'sphere',
                 int(mujoco.mjtGeom.mjGEOM_CAPSULE): 'capsule', int(mujoco.mjtGeom.mjGEOM_CYLINDER): 'cylinder',
                 int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 'ellipsoid', int(mujoco.mjtGeom.mjGEOM_PLANE): 'plane'}
        m = self.model
        for body in self.body_ids:
            geoms = []
            for gid in range(m.ngeom):
                if m.geom_bodyid[gid] != body or m.geom_group[gid] == 3:
                    continue
                kind = types.get(int(m.geom_type[gid]))
                pos, size = m.geom_pos[gid].copy(), m.geom_size[gid].copy()
                mesh_file = None
                if kind is None:
                    candidate = f'aloha-mesh-{m.geom_dataid[gid]}.stl'
                    if (Path(__file__).resolve().parent / 'web/assets/meshes' / candidate).is_file():
                        kind, mesh_file = 'mesh', candidate
                    else:
                        kind = 'box'
                        pos += m.geom_aabb[gid, :3]
                        size = m.geom_aabb[gid, 3:]
                geoms.append(GeomDesc(type=kind, size=tuple(size), pos=tuple(pos),
                                     quat=tuple(m.geom_quat[gid]), rgba=tuple(m.geom_rgba[gid]), mesh_file=mesh_file))
            result.append(RolloutBody(name=m.body(body).name or 'world', geoms=geoms))
        return result

    def attach(self, side):
        d, m = self.data, self.model
        distance = float(np.linalg.norm(self.site(side) - d.xpos[self.obj]))
        if distance > .055 or not self.closed[side]:
            return False
        grip = self.grips[side]
        rotation = d.xmat[grip].reshape(3, 3)
        relpos = rotation.T @ (d.xpos[self.obj] - d.xpos[grip])
        relrot = rotation.T @ d.xmat[self.obj].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, relrot.ravel())
        m.eq_data[self.eqs[side], 3:6] = relpos
        m.eq_data[self.eqs[side], 6:10] = quat
        d.eq_active[self.eqs[side]] = True
        self.hold_rotations[side] = d.site_xmat[self.sites[side]].reshape(3, 3).copy()
        self.tasks[side].set_orientation_cost(.15)
        self.holds[side] = self.grasped[side] = True
        return True

    def detach(self, side):
        self.data.eq_active[self.eqs[side]] = False
        self.holds[side] = False
        self.closed[side] = False

    def step(self, phase, targets, duration):
        m, d = self.model, self.data
        starts = {side: self.site(side) for side in targets}
        n = max(1, int(math.ceil(duration / .02)))
        for k in range(n):
            self.config.update(d.qpos)
            fraction = (k + 1) / n
            fraction = fraction * fraction * (3 - 2 * fraction)
            for side, target in targets.items():
                point = starts[side] + fraction * (np.asarray(target) - starts[side])
                yaw = math.radians(self.plan.approach_yaw_deg)
                if side == 'right':
                    yaw += math.pi + math.radians(self.plan.handover_yaw_offset_deg)
                rotation = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                                     [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.]])
                if self.holds[side]:
                    rotation = self.hold_rotations[side]
                self.tasks[side].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotation), point))
            velocity = mink.solve_ik(self.config, [*self.tasks.values(), self.posture], .02,
                                     solver='daqp', damping=1e-4, limits=self.limits)
            self.config.integrate_inplace(velocity, .02)
            d.ctrl[:] = self.config.q[self.act_qpos]
            for side in self.tasks:
                d.ctrl[m.actuator(f'{side}/gripper').id] = .002 if self.closed[side] else .035
            for _ in range(10):
                mujoco.mj_step(m, d)
                geom_id = m.geom(f'{self.scene.task_object}_geom').id
                z_axis = d.geom_xmat[geom_id].reshape(3, 3)[2]
                if self.object_spec.kind == 'cylinder':
                    support = self.object_spec.size_m[0] * np.linalg.norm(z_axis[:2]) + self.object_spec.size_m[1] / 2 * abs(z_axis[2])
                else:
                    support = float(np.dot(np.abs(z_axis), np.asarray(self.object_spec.size_m) / 2))
                self.min_z = min(self.min_z, float(d.geom_xpos[geom_id, 2] - support))
                self.record_contacts(phase)
            self.record_clearance(phase)
            if k % 2 == 0:
                self.frames.append([float(v) for body in self.body_ids for v in [*d.xpos[body], *d.xquat[body]]])
            if not np.isfinite(d.qpos).all():
                raise SimError('MuJoCo produced a non-finite state')

    def record_contacts(self, phase):
        """Measure the physical state, independently of the IK clearance constraint."""
        m, d = self.model, self.data
        obstacles = set(self.obstacle_geoms)
        robots = set(self.robot_geoms)
        for index, contact in enumerate(d.contact):
            a, b = int(contact.geom1), int(contact.geom2)
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, index, force)
            magnitude = float(np.linalg.norm(force[:3]))
            if (m.geom(a).name == 'desk' and b in robots) or (m.geom(b).name == 'desk' and a in robots):
                self.max_contact = max(self.max_contact, magnitude)
            pair = (a in obstacles and b in robots) or (b in obstacles and a in robots)
            held_object = (a in obstacles and b == self.task_geom) or (b in obstacles and a == self.task_geom)
            penetration = max(0.0, -float(contact.dist))
            if pair:
                self.max_obstacle_contact = max(self.max_obstacle_contact, magnitude)
                self.max_arm_penetration = max(self.max_arm_penetration, penetration)
            if held_object:
                self.max_object_penetration = max(self.max_object_penetration, penetration)
            violation = ((pair and (magnitude > config.MAX_ARM_OBSTACLE_CONTACT_FORCE_N
                                    or penetration > config.MAX_ARM_OBSTACLE_PENETRATION_M))
                         or (held_object and penetration > config.MAX_OBJECT_OBSTACLE_PENETRATION_M))
            if violation and self.failure is None:
                self.failure = phase
            if violation and len(self.collision_events) < 20:
                self.collision_events.append({'phase': phase, 'time_s': float(d.time),
                    'bodies': [m.body(m.geom_bodyid[g]).name for g in [a, b]],
                    'penetration_m': penetration, 'force_n': magnitude})

    def record_clearance(self, phase):
        for robot in self.robot_geoms:
            for obstacle in self.obstacle_geoms:
                distance = mujoco.mj_geomDistance(self.model, self.data, robot, obstacle, .10, None)
                self.min_obstacle_distance = min(self.min_obstacle_distance, float(distance))

    def move(self, phase, targets):
        distance = max(np.linalg.norm(np.asarray(point) - self.site(side)) for side, point in targets.items())
        self.step(phase, targets, max(.4, float(distance) / self.plan.max_speed_m_s * 1.5))
        self.step(phase, targets, .3)

    def snapshot(self, label, phase):
        if self.renderer is None:
            return
        for camera in ['overhead', 'front', 'left_wrist', 'right_wrist']:
            self.renderer.update_scene(self.data, camera=camera)
            path = f'{label}-{camera}.png'
            Image.fromarray(self.renderer.render()).save(self.out / path)
            self.keyframes.append(Keyframe(label=f'{label}_{camera}', phase=phase,
                                           t_s=float(self.data.time), camera=camera, path=path))

    def run(self):
        p = self.plan
        donor = getattr(self.scene, 'pickup_arm', 'left')
        receiver = 'right' if donor == 'left' else 'left'
        donor_sign = -1 if donor == 'left' else 1
        receiver_sign = -donor_sign
        center = self.data.xpos[self.obj].copy()
        grasp = center + [0, 0, p.grasp_height_offset_m]
        pre = grasp + [donor_sign * p.pregrasp_clearance_m, 0, .035]
        self.move('approach', {donor: pre})
        self.move('grasp', {donor: grasp})
        self.closed[donor] = True
        if not self.attach(donor):
            self.failure = 'grasp'
        self.step('grasp', {}, p.gripper_close_delay_s + .1)
        self.snapshot('grasp', 'grasp')
        self.move('lift', {donor: grasp + [0, 0, p.lift_height_m]})
        self.snapshot('lift', 'lift')
        handover = np.asarray(p.handover_point_m)
        self.move('handover', {donor: handover + [0, 0, .02],
                              receiver: handover + [receiver_sign * p.pregrasp_clearance_m, 0, -.02]})
        receiver_target = self.data.xpos[self.obj].copy() + [0, 0, -.025]
        self.move('handover', {receiver: receiver_target})
        self.closed[receiver] = True
        receiver_held = self.attach(receiver)
        self.step('handover', {}, p.gripper_close_delay_s)
        self.step('handover', {}, p.release_delay_s)
        self.detach(donor)
        self.step('handover', {}, .2)
        self.transferred = bool(self.grasped[donor] and receiver_held and self.holds[receiver] and not self.holds[donor]
                                and np.linalg.norm(self.site(receiver) - self.data.xpos[self.obj]) < .07)
        if not self.transferred and self.failure is None:
            self.failure = 'handover'
        self.snapshot('handover', 'handover')
        offset = self.data.xpos[self.obj] - self.site(receiver)
        goal_center = np.asarray(self.scene.target_m) + [0, 0, self.object_spec.size_m[-1] / 2 + p.release_height_m]
        self.move('place', {donor: handover + [donor_sign * .16, 0, .08],
                           receiver: goal_center - offset + [0, 0, .12]})
        self.move('place', {receiver: goal_center - offset})
        self.detach(receiver)
        self.snapshot('release', 'place')
        self.move('settle', {receiver: self.site(receiver) + [receiver_sign * .12, 0, .12]})
        self.step('settle', {}, 1.5)
        self.snapshot('final', 'settle')


def run_attempt(scene: SceneSpec, plan: HandoverPlan, *, attempt: int, out_dir: Path,
                render: bool = True, seed: int = 0) -> SimAttempt:
    start = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # No random perturbation in a nominal run; seed is reserved for explicit sweep inputs.
    runner = _Attempt(scene, plan, out_dir, render)
    try:
        runner.run()
        d, m = runner.data, runner.model
        position = d.xpos[runner.obj].copy()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, runner.obj, velocity, 0)
        metrics = AttemptMetrics(can_final_pos_m=tuple(position),
            can_final_xy_error_m=float(np.linalg.norm(position[:2] - np.asarray(scene.target_m)[:2])),
            can_min_z_m=runner.min_z, can_final_speed_m_s=float(np.linalg.norm(velocity[3:])),
            left_grasped=runner.grasped['left'], right_grasped=runner.grasped['right'],
            handover_completed=runner.transferred, max_arm_table_contact_force_n=runner.max_contact,
            max_arm_obstacle_contact_force_n=runner.max_obstacle_contact,
            max_arm_obstacle_penetration_m=runner.max_arm_penetration,
            max_object_obstacle_penetration_m=runner.max_object_penetration,
            grasp_assist_used=any(runner.grasped.values()), failure_phase=runner.failure,
            sim_time_s=float(d.time), wall_time_ms=(time.monotonic() - start) * 1000)
        (out_dir / 'collision-evidence.json').write_text(json.dumps({
            'method': 'MuJoCo contacts at 500 Hz; IK clearance constraint at 50 Hz',
            'ik_clearance_m': .02, 'min_arm_obstacle_distance_m': runner.min_obstacle_distance,
            'obstacles': [obj.name for obj in scene.objects if not obj.graspable],
            'events': runner.collision_events,
        }, indent=2))
        log = RolloutLog(fps=25, bodies=runner.bodies, frames=runner.frames)
        (out_dir / 'rollout.json').write_text(log.model_dump_json())
        result = SimAttempt(attempt=attempt, metrics=metrics, keyframes=runner.keyframes,
                            rollout_path='rollout.json', mjcf_path='scene.xml', warnings=runner.warnings)
        (out_dir / 'sim-attempt.json').write_text(result.model_dump_json(indent=2))
        return result
    finally:
        if runner.renderer is not None:
            runner.renderer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('--out', type=Path, default=Path('runs/manual/attempt-1'))
    parser.add_argument('--no-render', action='store_true')
    args = parser.parse_args()
    body = json.loads(args.scene.read_text())
    scene = SceneSpec.model_validate(body['scene'])
    plan = HandoverPlan.model_validate(body['plan'])
    print(run_attempt(scene, plan, attempt=1, out_dir=args.out, render=not args.no_render).model_dump_json(indent=2))
