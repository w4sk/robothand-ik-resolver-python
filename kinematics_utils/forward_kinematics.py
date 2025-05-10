import numpy as np


class FKResolver:
    def __init__(self, robot_arm_lengths):
        self.lengths = robot_arm_lengths

    def _rot_z(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def _rot_y(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array(
            [
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ]
        )

    def _trans(self, x=0, y=0, z=0):
        return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def calc_forward_kinematics(self, joint_angles, is_degrees=True):
        if is_degrees:
            joint_angles = [angle * np.pi / 180 for angle in joint_angles]

        H1 = np.array(
            [
                [1, 0, 0, 0],
                [0, 2, 0, self.lengths[0]],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transforms = [
            (self._rot_y(joint_angles[0]), self._trans(0, self.lengths[1], -self.lengths[2] - self.lengths[5])),
            (self._rot_z(joint_angles[1]), self._trans(0, self.lengths[3], 0)),
            (self._rot_z(joint_angles[2]), self._trans(0, self.lengths[4], 0)),
            (self._rot_z(joint_angles[3]), self._trans(0, self.lengths[6], 0)),
            (self._rot_y(joint_angles[4]), self._trans(0, self.lengths[7], -self.lengths[8])),
        ]

        H = H1
        positions = []
        base_pos = np.array([0, 0, 0, 1])
        for R, T in transforms:
            H = np.dot(H, np.dot(R, T))
            pos = np.dot(H, base_pos)
            positions.append({"x": pos[0], "y": pos[1], "z": pos[2]})

        j1_pos = np.dot(H1, base_pos)
        positions.insert(0, {"x": j1_pos[0], "y": j1_pos[1], "z": j1_pos[2]})

        return positions


if __name__ == "__main__":
    length = [0.125, 0.093, 0.05, 0.28, 0.225, 0.0655, 0.064, 0.056, 0.0845]
    fk = FKResolver(length)
    wknodes = fk.calc_forward_kinematics(joint_angles=[0, 0, 0, 0, 0], is_degrees=True)
    for i, wknode in enumerate(wknodes):
        print(f"joint{i} x: {wknode['x']}, y: {wknode['y']}, z: {wknode['z']}")
