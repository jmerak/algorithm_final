import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
import pulp
import random


# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def generate_locations(j, k, area_size=(100, 100), max_distance=10):
    """
    生成j个配送中心和k个卸货点的随机位置信息，确保任意配送中心到卸货点的距离不超过max_distance。

    参数：
    j (int): 配送中心的数量。
    k (int): 卸货点的数量。
    area_size (tuple): 图的区域大小，默认为 (100, 100)。
    max_distance (float): 配送中心到卸货点的最大距离，默认为20公里。

    返回：
    tuple: 包含配送中心和卸货点的位置信息。
    """

    def distance(point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    while True:
        # 生成配送中心和卸货点的位置信息
        distribution_centers = np.random.rand(j, 2) * area_size
        delivery_points = np.random.rand(k, 2) * area_size

        # 检查所有配送中心到卸货点的距离是否满足最大距离要求
        valid = True
        for dc in distribution_centers:
            for dp in delivery_points:
                if distance(dc, dp) > max_distance:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            break

    return distribution_centers, delivery_points


# 计算两点之间的距离
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def plot_locations(distribution_centers, delivery_points):
    """
    绘制配送中心和卸货点的位置图。

    参数：
    distribution_centers (ndarray): 配送中心的位置信息。
    delivery_points (ndarray): 卸货点的位置信息。
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(distribution_centers[:, 0], distribution_centers[:, 1], c='red', marker='s', label='配送中心')
    plt.scatter(delivery_points[:, 0], delivery_points[:, 1], c='blue', marker='o', label='卸货点')

    for i, (x, y) in enumerate(distribution_centers):
        plt.text(x, y, f'DC{i + 1}', fontsize=12, ha='right')

    for i, (x, y) in enumerate(delivery_points):
        plt.text(x, y, f'DP{i + 1}', fontsize=12, ha='right')

    plt.title('配送中心和卸货点位置图')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_orders(k, max_orders=5):
    """
    生成随机订单，每个订单有优先级。
    """
    orders = []
    for i in range(k):
        num_orders = random.randint(0, max_orders)
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            orders.append((i, priority))
    return orders


def linear_programming(distribution_centers, delivery_points, orders, max_distance=20, n=3):
    """
    使用线性规划进行无人机路径规划。
    """
    j = len(distribution_centers)
    k = len(delivery_points)

    # 计算距离矩阵
    distance_matrix = np.zeros((j, k))
    for i in range(j):
        for l in range(k):
            distance_matrix[i][l] = calculate_distance(distribution_centers[i], delivery_points[l])

    # 创建线性规划问题
    lp_problem = pulp.LpProblem("Drone_Delivery_Optimization", pulp.LpMinimize)

    # 创建决策变量
    x = pulp.LpVariable.dicts("x", [(i, l) for i in range(j) for l in range(k)], cat='Binary')

    # 目标函数：最小化总配送路径
    lp_problem += pulp.lpSum([x[i, l] * distance_matrix[i][l] for i in range(j) for l in range(k)])

    # 约束条件：每个卸货点必须被配送
    for order in orders:
        l = order[0]
        lp_problem += pulp.lpSum([x[i, l] for i in range(j)]) >= 1

    # 约束条件：无人机一次最多携带n个物品
    for i in range(j):
        lp_problem += pulp.lpSum([x[i, l] for l in range(k)]) <= n

    # 约束条件：无人机总飞行距离不超过max_distance
    for i in range(j):
        lp_problem += pulp.lpSum([x[i, l] * distance_matrix[i][l] for l in range(k)]) <= max_distance

    # 求解线性规划问题
    lp_problem.solve()

    # 输出结果
    result = []
    for v in lp_problem.variables():
        if v.varValue > 0:
            result.append((v.name, v.varValue))

    return result, pulp.value(lp_problem.objective)


def simulate_delivery(j, k, T, max_orders=5, simulation_time=60, area_size=(10, 10), max_distance=10, n=3):
    """
    模拟无人机配送路径规划。
    """
    distribution_centers, delivery_points = generate_locations(j, k, area_size, max_distance)
    total_distance = 0

    for t in range(0, simulation_time, T):
        orders = generate_orders(k, max_orders)
        print(f"时间 {t} 分钟, 生成订单: {orders}")

        result, distance = linear_programming(distribution_centers, delivery_points, orders, max_distance, n)
        total_distance += distance

        print(f"决策结果: {result}")
        print(f"当前时间步长的总配送路径: {distance}")

    print(f"总配送路径: {total_distance}")
    return distribution_centers, delivery_points


# 示例使用
j = 10  # 配送中心的数量
k = 20  # 卸货点的数量
n = 3  # 无人机一次最多携带的物品数量
area_size = (10, 10)  # 区域大小
max_distance = 10  # 最大飞行距离为20公里
speed = 60  # 无人机速度为60公里/小时
T = 10 # 时间步长（分钟）
simulation_time = 60 * 24# 总模拟时间（分钟）

# 生成并绘制地图
distribution_centers, delivery_points = simulate_delivery(j, k, T, simulation_time=simulation_time, area_size=area_size)

plot_locations(distribution_centers, delivery_points)

