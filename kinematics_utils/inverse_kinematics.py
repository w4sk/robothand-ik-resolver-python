import numpy as np


class IKSolver:
    def __init__(self, robot_arm_lengths, initial_position, is_rghit_hand=True):
        self.lengths = robot_arm_lengths
        self.xy_fab_length = [robot_arm_lengths[3], robot_arm_lengths[4], robot_arm_lengths[6] + robot_arm_lengths[7]]
        self.is_right_hand = is_rghit_hand
        self.last_position = initial_position
        self.last_xy_nodes = np.array([[0, 0.218], [-1, 0.5], [-1, 0.723]])

    def calc_inverse_kinematics(self, goal_position, goal_quaternion, debug=False):
        def _calc_distance(pos_1, pos_2):
            return np.sqrt(np.sum((np.array(pos_1) - np.array(pos_2)) ** 2))

        def _quaternion_to_matrix(q):
            x, y, z, w = q
            matrix = np.array(
                [
                    [2 * w**2 + 2 * x**2 - 1, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                    [2 * x * y + 2 * z * w, 2 * w**2 + 2 * y**2 - 1, 2 * y * z - 2 * x * w],
                    [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 2 * w**2 + 2 * z**2 - 1],
                ]
            )

            return matrix

        def _normalize_angle(angle):
            angle = angle % (2 * np.pi)
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle <= -np.pi:
                angle += 2 * np.pi
            return angle

        def _calc_local_unit_vectors(H):
            """
            H: 3x3の回転行列 (numpy.ndarray)
            戻り値: 辞書で'x_vector', 'y_vector', 'z_vector'が正規化された3次元ベクトル(np.array)
            """
            # 3x3回転行列を4x4に拡張（右下を1に）
            H4 = np.eye(4)
            H4[:3, :3] = H

            # 各軸方向の平行移動行列
            TX = np.array(
                [
                    [1, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            TY = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )
            TZ = np.array(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ]
            )

            origin = np.array([[0], [0], [0], [1]])

            x = H4 @ TX @ origin
            y = H4 @ TY @ origin
            z = H4 @ TZ @ origin

            # ベクトル成分を取り出し、原点からのベクトルに変換
            x_vector = x[:3, 0] - H4[:3, 3]
            y_vector = y[:3, 0] - H4[:3, 3]
            z_vector = z[:3, 0] - H4[:3, 3]

            # 正規化
            x_vector /= np.linalg.norm(x_vector)
            y_vector /= np.linalg.norm(y_vector)
            z_vector /= np.linalg.norm(z_vector)

            return {
                "x_vector": x_vector,
                "y_vector": y_vector,
                "z_vector": z_vector,
            }

        def _calc_j1_rotation_j2_position(j6back_pos):
            xz_base_j6back_length = _calc_distance(np.array([0, 0, 0]), [j6back_pos[0], 0, j6back_pos[2]])
            xz_j1_j2_length = self.lengths[2] + self.lengths[5]
            xz_j2_j6back_length = np.sqrt(max(np.finfo(float).eps, xz_base_j6back_length**2 - xz_j1_j2_length**2))
            ratio = np.clip(xz_j1_j2_length / xz_j2_j6back_length, -1, 1)
            xz_j6back_j1_j2_angle = np.arccos(ratio)
            xz_j6back_angle = np.arctan2(-j6back_pos[0], -j6back_pos[2])

            if self.is_right_hand:
                j1_rotation = _normalize_angle(xz_j6back_angle - xz_j6back_j1_j2_angle)
            else:
                j1_rotation = _normalize_angle(xz_j6back_angle + xz_j6back_j1_j2_angle)
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

        def _calc_j3_position_j2_j3_rotation(j2_position, j4_position):
            def _convert_local_to_global(j2_position, j4_position, j2j3j4_positions):
                j2_j4_vector = j4_position - j2_position
                j2_j4_vector /= np.linalg.norm(j2_j4_vector)

                x_global = j2_j4_vector[0] * j2j3j4_positions[1][0] + j2_position[0]
                y_global = j2j3j4_positions[1][1]
                z_global = j2_j4_vector[0] * j2j3j4_positions[1][0] + j2_position[2]

                return np.array([x_global, y_global, z_global])

            j4_pos_for_fabrik = np.array(
                [
                    _calc_distance([j2_position[0], 0, j2_position[2]], [j4_position[0], 0, j4_position[2]]),
                    j4_position[1],
                ]
            )
            j2_pos_for_fabrik = np.array([0, j2_position[1]])

            j2j3j4_positions = self.xy_fabrik(j4_pos_for_fabrik, j2_pos_for_fabrik)

            j3_position = _convert_local_to_global(j2_position, j4_position, j2j3j4_positions)

            local_j2j3_vector = np.array(j2j3j4_positions[1]) - np.array(j2j3j4_positions[0])
            j2_rotation = np.arctan2(local_j2j3_vector[0], local_j2j3_vector[1])

            local_j3j4_vector = np.array(j2j3j4_positions[2]) - np.array(j2j3j4_positions[1])
            j3_rotation = np.arctan2(local_j3j4_vector[0], local_j3j4_vector[1]) - j2_rotation

            return j3_position, j2_rotation, j3_rotation

        def _calc_j6_position(j6back_pos, local_unit_vectors):
            return j6back_pos + local_unit_vectors["z_vector"] * -self.lengths[8]

        def _calc_j4_rotation(j4_position, j5_position, j2_rotation, j3_rotation, unit_direction_vector):
            xz_base_j4_length = _calc_distance([j4_position[0], 0, j4_position[2]], [0, 0, 0])
            xz_base_j5_length = _calc_distance([j5_position[0], 0, j5_position[2]], [0, 0, 0])

            xz_j1_j2_length = self.lengths[2] + self.lengths[5]
            xz_j2_j4_rength = np.sqrt(xz_base_j4_length**2 - xz_j1_j2_length**2)
            xz_j2_j5_length = np.sqrt(xz_base_j5_length**2 - xz_j1_j2_length**2)

            direction_multiplier = -1 if xz_j2_j5_length > xz_j2_j4_rength else 1
            xy_j5_j4_vector = np.array(
                [
                    direction_multiplier
                    * self.lengths[6]
                    * np.sqrt(unit_direction_vector[0] ** 2 - unit_direction_vector[2] ** 2),
                    -unit_direction_vector[1] * self.lengths[6],
                ]
            )

            j4_rotation = np.arctan2(-xy_j5_j4_vector[0], -xy_j5_j4_vector[1]) - j2_rotation - j3_rotation
            return j4_rotation

        def _calc_j5_rotation(j2_position, local_unit_vectors, unit_direction_vector):
            xz_base_j2_vector = np.array([j2_position[0], 0, j2_position[2]])
            xz_base_j2_vector /= np.linalg.norm(xz_base_j2_vector)
            local_z_vector = np.cross(xz_base_j2_vector, unit_direction_vector)
            local_z_vector /= np.linalg.norm(local_z_vector)
            A = np.array(
                [
                    [xz_base_j2_vector[0], local_z_vector[0], unit_direction_vector[0]],
                    [xz_base_j2_vector[1], local_z_vector[1], unit_direction_vector[1]],
                    [xz_base_j2_vector[2], local_z_vector[2], unit_direction_vector[2]],
                ]
            )

            b = np.array(
                [
                    [-local_unit_vectors["z_vector"][0]],
                    [-local_unit_vectors["z_vector"][1]],
                    [-local_unit_vectors["z_vector"][2]],
                ]
            )

            rotation_matrix_solution = np.linalg.solve(A, b)
            return np.arctan2(-rotation_matrix_solution[1, 0], rotation_matrix_solution[0, 0])

        def _calc_j6_rotation(local_unit_vectors, unit_direction_vector):
            j6_rotation = np.arccos(np.clip(np.dot(local_unit_vectors["y_vector"], unit_direction_vector), -1.0, 1.0))

            right_hand_rule_check = np.dot(
                np.cross(local_unit_vectors["z_vector"], unit_direction_vector), local_unit_vectors["y_vector"]
            )

            if right_hand_rule_check < 0:
                j6_rotation *= -1
            elif j6_rotation == 0 or np.isnan(j6_rotation):
                j6_rotation = 0.0

            return j6_rotation

        joint_rotation_results = {
            "j1": 0,
            "j2": 0,
            "j3": 0,
            "j4": 0,
            "j5": 0,
            "j6": 0,
        }

        joint_position_results = {
            "j1": np.array([0, self.lengths[0], 0]),
            "j2": np.array([0, 0, 0]),
            "j3": np.array([0, 0, 0]),
            "j4": np.array([0, 0, 0]),
            "j5": np.array([0, 0, 0]),
            "j6": np.array([goal_position[0], goal_position[1], goal_position[2]]),
        }

        goal_position = goal_position + self.last_position

        transformation_matrix = _quaternion_to_matrix(goal_quaternion)
        local_unit_vectors = _calc_local_unit_vectors(transformation_matrix)

        j6back_pos = np.array(
            [
                goal_position[0] + local_unit_vectors["z_vector"][0] * self.lengths[8],
                goal_position[1] + local_unit_vectors["z_vector"][1] * self.lengths[8],
                goal_position[2] + local_unit_vectors["z_vector"][2] * self.lengths[8],
            ]
        )

        j1_rotation, j2_position = _calc_j1_rotation_j2_position(j6back_pos)
        joint_rotation_results["j1"] = j1_rotation
        joint_position_results["j2"] = j2_position

        xz_base_j2_vector = np.array([j2_position[0], 0, j2_position[2]])

        unit_direction_vector = np.cross(xz_base_j2_vector, local_unit_vectors["z_vector"])
        unit_direction_vector /= np.linalg.norm(unit_direction_vector)
        if unit_direction_vector[1] < 0:
            unit_direction_vector *= -1

        j4_position, j5_position = _calc_j4_j5_position(j6back_pos, self.last_position, unit_direction_vector)
        joint_position_results["j4"] = j4_position
        joint_position_results["j5"] = j5_position

        j3_position, j2_rotation, j3_rotation = _calc_j3_position_j2_j3_rotation(j2_position, j4_position)
        joint_position_results["j3"] = j3_position
        joint_rotation_results["j2"] = j2_rotation
        joint_rotation_results["j3"] = j3_rotation

        j6_position = _calc_j6_position(j6back_pos, local_unit_vectors)
        joint_position_results["j6"] = j6_position

        j4_rotation = _calc_j4_rotation(j4_position, j5_position, j2_rotation, j3_rotation, unit_direction_vector)
        joint_rotation_results["j4"] = j4_rotation

        j5_rotation = _calc_j5_rotation(j2_position, local_unit_vectors, unit_direction_vector)
        joint_rotation_results["j5"] = j5_rotation

        j6_rotation = _calc_j6_rotation(local_unit_vectors, unit_direction_vector)
        joint_rotation_results["j6"] = j6_rotation

        if debug:
            print(f"[IK DEBUG] joint positions: {joint_position_results}")
            print(f"[IK DEBUG] joint rotations: {joint_rotation_results}")

        return joint_position_results, joint_rotation_results

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


if __name__ == "__main__":
    robot_arm_lengths = [0.0353, 0.0353, 0.06244, 0.1104, 0.096, 0, 0.01488, 0.01488, 0.0456]
    ik_solver = IKSolver(robot_arm_lengths=robot_arm_lengths)
    # ik_solver.calc_inverse_kinematics()
