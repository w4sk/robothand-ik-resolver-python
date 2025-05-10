import numpy as np


class IKSolver:
    def __init__(self, robot_arm_lengths, is_rghit_hand=True):
        self.lengths = robot_arm_lengths
        self.xy_fab_length = [robot_arm_lengths[3], robot_arm_lengths[4], robot_arm_lengths[6] + robot_arm_lengths[7]]
        self.is_right_hand = is_rghit_hand
        self.last_xy_nodes = np.array([0, 0.218], [-1, 0.5], [-1, 0.723])

    def calc_inverse_kinematics(self):
        def _calc_distance(pos_1, pos_2):
            return np.sqrt(np.sum((np.array(pos_1) - np.array(pos_2)) ** 2))

        def _calc_j1_rotation_j2_position(j6back_pos):
            xz_base_j6back_length = _calc_distance(np.array([0, 0, 0]), j6back_pos)
            xz_j1_j2_length = self.lengths[2] + self.lengths[5]
            xz_j2_j6back_length = np.sqrt(max(np.finfo(float).eps, xz_base_j6back_length**2 - xz_j1_j2_length**2))
            ratio = np.clip(xz_j1_j2_length / xz_j2_j6back_length, -1, 1)
            xz_j6back_j1_j2_angle = np.arccos(ratio)
            xz_j6back_angle = np.arctan2(-j6back_pos[0], -j6back_pos[2])

            if self.is_right_hand:
                j1_rotation = self.normalize_angle(xz_j6back_angle - xz_j6back_j1_j2_angle)
            else:
                j1_rotation = self.normalize_angle(xz_j6back_angle + xz_j6back_j1_j2_angle)

            j2_position = np.array(
                [
                    -xz_j1_j2_length * np.sin(j1_rotation),
                    self.lengths[0] + self.lengths[1],
                    -xz_j1_j2_length * np.cos(j1_rotation),
                ]
            )

            return j1_rotation, j2_position

        def _is_first_position_closer_to_target(pos1, pos2, target):
            dist1 = np.linalg.norm(pos1 - target)
            dist2 = np.linalg.norm(pos2 - target)
            return dist1 < dist2

        def _calc_j4_j5_position(j6back_pos, last_pos, unit_direction_vector):
            j5_positive_vector = j6back_pos + unit_direction_vector * self.lengths[7]
            j5_negative_vector = j6back_pos + unit_direction_vector * (-self.lengths[7])

            if _is_first_position_closer_to_target(j5_positive_vector, j5_negative_vector, last_pos):
                j5_position = j5_positive_vector
                j4_position = j6back_pos + unit_direction_vector * (self.lengths[6] + self.lengths[7])
            else:
                j5_position = j5_negative_vector
                j4_position = j6back_pos + unit_direction_vector * (-(self.lengths[6] + self.lengths[7]))

            return j4_position, j5_position

        def _calc_j3_position_j2_j3_rotation(joint_position_results):
            def convert_local_to_global(joint_positon_results, j2j3j4_positions):
                j2 = joint_positon_results["j2"]
                j4 = joint_positon_results["j4"]

                j2_j4_vector = j4 - j2
                j2_j4_vector /= np.linalg.norm(j2_j4_vector)

                x_global = j2_j4_vector[0] * j2j3j4_positions[1][0] + j2[0]
                y_global = j2j3j4_positions[1][1]
                z_global = j2_j4_vector[0] * j2j3j4_positions[1][0] + j2[2]

                return np.array([x_global, y_global, z_global])

            j2 = joint_position_results["j2"]
            j4 = joint_position_results["j4"]

            j4_pos_for_fabrik = np.array(_calc_distance([j2[0], 0, j2[2]], [j4[0], 0, j4[2]]), j4[1])
            j2_pos_for_fabrik = np.array([0, j2[1]])

            j2j3j4_positions = self.xy_fabrik(j4_pos_for_fabrik, j2_pos_for_fabrik)

            j3_position = convert_local_to_global(joint_position_results, j2j3j4_positions)

            local_j2j3_vector = np.array(j2j3j4_positions[1]) - np.array(j2j3j4_positions[0])
            j2_rotation = np.arctan2(local_j2j3_vector[1], local_j2j3_vector[0])

            local_j3j4_vector = np.array(j2j3j4_positions[2]) - np.array(j2j3j4_positions[1])
            j3_rotation = np.arctan2(local_j3j4_vector[1], local_j3j4_vector[0]) - j2_rotation

            return j3_position, j2_rotation, j3_rotation
        
        def _calc_j6_position(j6back_pos, local_unit_vectors):
            return j6back_pos + local_unit_vectors[0] * self.lengths[8]

    def xy_fabrik(self, target, start):
        def xy_get_point(t_num, s_num, xy_nodes):
            t = xy_nodes[t_num]
            s = xy_nodes[s_num]

            d = np.linalg.norm(t - s)
            if d < np.finfo(float).eps:
                dx, dy = 0.0, 0.0
            else:
                dx, dy = (t - s) / d

            if t_num > s_num:
                p = t - np.array([dx, dy]) * self.xy_fab_length[s_num]
            else:
                p = t - np.array([dx, dy]) * self.xy_fab_length[t_num]

            return p

        xy_nodes = self.last_xy_nodes

        nan_mask = np.isnan(xy_nodes)
        xy_nodes[nan_mask] = 0.0

        length = len(xy_nodes) - 1

        for _ in range(50):
            xy_nodes[length] = target
            for j in range(1, length + 1):
                xy_nodes[length - j] = xy_get_point(length - j + 1, length - j, xy_nodes)

            xy_nodes[0] = start
            for j in range(1, length + 1):
                xy_nodes[j] = xy_get_point(j - 1, j, xy_nodes)

        self.last_xy_nodes = xy_nodes.copy()
        return xy_nodes
