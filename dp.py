import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def generate_map1(num_centers, num_points, map_size, max_distance):
    centers = [
        (0, (max_distance/2, max_distance/2)),
        (1, (max_distance/2, map_size-max_distance/2)),
        (2, (map_size-max_distance/2, max_distance/2)),
        (3, (map_size-max_distance/2, map_size-max_distance/2)),
        (4, (map_size / 2, map_size / 2))
    ]

    points = []
    point_id = num_centers

    for center in centers:
        center_id, center_coord = center
        num_points_per_center = num_points // num_centers
        for _ in range(num_points_per_center):
            while True:
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, max_distance / 2)
                point_x = center_coord[0] + radius * np.cos(angle)
                point_y = center_coord[1] + radius * np.sin(angle)

                if 0 <= point_x <= map_size and 0 <= point_y <= map_size:
                    point = (point_id, (point_x, point_y))
                    points.append(point)
                    point_id += 1
                    break

    return centers, points

def plot_map(centers, points, best_route_H, best_route_S, map_size, max_distance):
    plt.figure(figsize=(12, 12))

    # Plot delivery centers
    centers_x, centers_y = zip(*[center[1] for center in centers])
    plt.scatter(centers_x, centers_y, c='blue', marker='s', s=100, label='配送中心')
    for i, txt in enumerate(range(len(centers))):
        plt.annotate(txt, (centers_x[i], centers_y[i]), fontsize=12, fontweight='bold', ha='right')

    # Plot service radius circles for centers
    for center in centers:
        circle = Circle(center[1], max_distance / 2, color='blue', alpha=0.1, linestyle='--')
        plt.gca().add_patch(circle)

    # Plot delivery points
    points_x, points_y = zip(*[point[1] for point in points])
    plt.scatter(points_x, points_y, c='green', marker='o', s=60, label='卸货点')
    for i, txt in enumerate(range(len(points))):
        plt.annotate(txt, (points_x[i], points_y[i]), fontsize=10, ha='right')

    # Plot best routes found by the algorithm
    best_route_H = [point - 1 for point in best_route_H]  # Convert to 0-based indexing
    best_route_S = [point - 1 for point in best_route_S]
    plt.plot(points_x[best_route_H + [best_route_H[0]]], points_y[best_route_H + [best_route_H[0]]], 'o-',
             color='red', linewidth=2, alpha=0.7)
    plt.plot(points_x[best_route_S + [best_route_S[0]]], points_y[best_route_S + [best_route_S[0]]], 'o-',
             color='orange', linewidth=2, alpha=0.7)

    plt.title('无人机配送路径规划', fontsize=16, fontweight='bold')
    plt.xlabel('X 坐标', fontsize=14)
    plt.ylabel('Y 坐标', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis([0, map_size, 0, map_size])
    plt.show()

# Example parameters
map_size = 100
num_centers = 5
num_points = 20
max_distance = 20

# Generate map
centers, points = generate_map1(num_centers, num_points, map_size, max_distance)

# Placeholder for best routes (to be replaced with actual algorithm results)
best_route_H = list(range(1, num_points + 1))  # Example: visiting points in order
best_route_S = list(range(num_points, 0, -1))  # Example: visiting points in reverse order

# Plot the map and routes
plot_map(centers, points, best_route_H, best_route_S, map_size, max_distance)
