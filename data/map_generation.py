import numpy as np
import random
import math


def generate_map1(num_centers, num_points, map_size, max_distance):
    # 在地图的四个角和中心生成配送中心
    centers = [
        (0, (max_distance / 2, max_distance / 2)),
        (1, (max_distance / 2, map_size - max_distance / 2)),
        (2, (map_size - max_distance / 2, max_distance / 2)),
        (3, (map_size - max_distance / 2, map_size - max_distance / 2)),
        (4, (map_size / 2, map_size / 2))
    ]

    points = []
    point_id = num_centers  # 编号从配送中心之后开始

    for center in centers:
        center_id, center_coord = center
        num_points_per_center = num_points // num_centers  # 平均分配给每个配送中心的卸货点数量

        # 确定性地生成卸货点，均匀分布在配送中心附近
        for i in range(num_points_per_center):
            angle = (2 * np.pi / num_points_per_center) * i
            radius = max_distance / 4  # 确定半径，以保证均匀分布在配送中心附近
            point_x = center_coord[0] + radius * np.cos(angle)
            point_y = center_coord[1] + radius * np.sin(angle)

            # 确保生成的点在地图范围内
            if 0 <= point_x <= map_size and 0 <= point_y <= map_size:
                point = (point_id, (point_x, point_y))
                points.append(point)
                point_id += 1

    # 计算所有点对之间的距离
    all_points = centers + points
    num_all_points = len(all_points)
    distance_matrix = np.zeros((num_all_points, num_all_points))

    for i in range(num_all_points):
        for j in range(i + 1, num_all_points):
            point1 = all_points[i][1]
            point2 = all_points[j][1]
            distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance

    return centers, points, distance_matrix


def generate_map(num_centers, num_points, map_size, max_distance):
    # 在地图的四个角和中心生成配送中心
    centers = [
        (0, (max_distance / 2, max_distance / 2)),
        (1, (max_distance / 2, map_size - max_distance / 2)),
        (2, (map_size - max_distance / 2, max_distance / 2)),
        (3, (map_size - max_distance / 2, map_size - max_distance / 2)),
        (4, (map_size / 2, map_size / 2))
    ]

    points = []
    point_id = num_centers  # 编号从配送中心之后开始

    for center in centers:
        center_id, center_coord = center
        num_points_per_center = num_points // num_centers  # 平均分配给每个配送中心的卸货点数量
        for _ in range(num_points_per_center):
            while True:
                # 在配送中心为圆心，最大飞行距离的一半为半径的范围内生成卸货点
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, max_distance / 2)
                point_x = center_coord[0] + radius * np.cos(angle)
                point_y = center_coord[1] + radius * np.sin(angle)

                # 确保生成的点在地图范围内
                if 0 <= point_x <= map_size and 0 <= point_y <= map_size:
                    point = (point_id, (point_x, point_y))
                    points.append(point)
                    point_id += 1
                    break

    # 计算所有点对之间的距离
    all_points = centers + points
    num_all_points = len(all_points)
    distance_matrix = np.zeros((num_all_points, num_all_points))

    for i in range(num_all_points):
        for j in range(i + 1, num_all_points):
            point1 = all_points[i][1]
            point2 = all_points[j][1]
            distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return centers, points, distance_matrix
