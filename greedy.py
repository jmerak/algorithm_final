import numpy as np
import random
from scipy.spatial.distance import euclidean
from collections import deque

# 定义配送中心和卸货点的坐标
distribution_centers = [(0, 0), (20, 20)]  # 示例坐标
delivery_points = [(5, 5), (10, 10), (15, 15), (25, 25)]  # 示例坐标

# 订单生成规则
t = 10  # 每隔 t 分钟生成订单
max_orders = 5  # 每次生成的最大订单数量

# 无人机属性
drone_speed = 60  # 公里/小时
max_load = 3  # 无人机最大携带量
max_flight_distance = 20  # 无人机最大飞行距离（单程）

# 订单队列
orders = deque()

# 生成订单
def generate_orders():
    new_orders = []
    for _ in range(random.randint(0, max_orders)):
        point = random.choice(delivery_points)
        priority = random.choices(['一般', '较紧急', '紧急'], [0.5, 0.3, 0.2])[0]
        new_orders.append((point, priority))
    orders.extend(new_orders)

# 计算路径长度
def calculate_path_length(path):
    total_length = 0
    for i in range(1, len(path)):
        total_length += euclidean(path[i-1], path[i])
    return total_length

# 贪婪算法规划路径
def plan_paths():
    paths = []
    current_orders = sorted(orders, key=lambda x: x[1], reverse=True)
    while current_orders:
        load = 0
        path = [random.choice(distribution_centers)]
        for order in current_orders[:]:
            if load < max_load:
                path.append(order[0])
                current_orders.remove(order)
                load += 1
        path.append(path[0])
        if calculate_path_length(path) <= max_flight_distance:
            paths.append(path)
        else:
            break
    return paths

# 模拟一天的订单生成和路径规划
def simulate_day():
    total_paths = []
    for _ in range(24 * 60 // t):
        generate_orders()
        paths = plan_paths()
        total_paths.extend(paths)
        orders.clear()  # 清空已处理的订单队列
    return total_paths

# 运行模拟
paths = simulate_day()
for i, path in enumerate(paths):
    print(f"Path {i+1}: {path}, Length: {calculate_path_length(path):.2f} km")
