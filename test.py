import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import mpl


# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 参数
num_centers = 5
num_points = 50
map_size = 10
t = 30
n = 5
max_distance = 20
speed = 60
time_limit = 24 * 60 // t
priority_weights = {'一般': 1, '较紧急': 2, '紧急': 3}
population_size = 500
generations = 10
mutation_rate = 0.1
crossover_rate = 0.9

# 生成确定性地图
def generate_deterministic_map():
    # 确定性的配送中心坐标
    centers = [
        (0, (2, 2)),
        (1, (2, 8)),
        (2, (8, 2)),
        (3, (8, 8)),
        (4, (5, 5))
    ]

    # 确定性的卸货点坐标
    points = [
        (5, (3, 4)),
        (6, (1, 5)),
        (7, (9, 6)),
        (8, (6, 5)),
        (9, (7, 4)),
        (10, (3, 9)),
        (11, (2, 7)),
        (12, (9, 1)),
        (13, (9, 9)),
        (14, (1, 4))
    ]

    return centers, points


# 生成配送中心和卸货点的坐标和编号
def generate_map(num_centers, num_points, map_size, max_distance):
    centers = [(i, (random.uniform(0, map_size), random.uniform(0, map_size))) for i in range(num_centers)]

    points = []
    point_id = num_centers  # 编号从配送中心之后开始
    for _ in range(num_points):
        while True:
            point = (point_id, (random.uniform(0, map_size), random.uniform(0, map_size)))
            if all(calculate_distance(center[1], point[1]) <= max_distance / 2 for center in centers):
                points.append(point)
                point_id += 1
                break

    return centers, points

# 距离计算
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 生成订单函数，使用模拟时间
def generate_orders(points):
    orders = []
    current_time = 0  # 模拟时间从0开始
    for point in points:
        num_orders = random.randint(1, 3)  # 每个卸货点都至少有一个订单
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            order_time = current_time
            if priority == '一般':
                deadline = current_time + 180  # 3小时
            elif priority == '较紧急':
                deadline = current_time + 90  # 1.5小时
            else:  # '紧急'
                deadline = current_time + 30  # 30分钟
            orders.append((point, priority, order_time, deadline))  # 添加订单信息
    return orders


def print_individual(individual):
    print("path：")
    for path in individual:
        path_str = " -> ".join(str(point[0]) for point in path)
        path_distance = sum(calculate_distance(path[i][1], path[i + 1][1]) for i in range(len(path) - 1))
        print(f"路径: {path_str}，长度: {path_distance:.2f}")


def initialize_population(centers, points, population_size, max_distance, n):
    population = []
    for _ in range(population_size):
        individual = [[] for _ in centers]  # 为每个配送中心初始化一条路径
        population.append([path for path in individual_fix(individual) if path])
    return population



def fitness(individual, orders, current_time):
    total_distance = 0
    total_time = 0
    penalty = 0  # 添加 penalty 变量
    for path in individual:
        path_distance = 0
        for i in range(len(path) - 1):
            path_distance += calculate_distance(path[i][1], path[i + 1][1])
        if path_distance > max_distance:
            penalty += 10000  # 如果路径超过最大飞行距离，给予较大的惩罚
        total_distance += path_distance

    # 计算订单时效性惩罚
    for path in individual:
        for order in path[1:-1]:  # 跳过起始和终点
            order_id = order[0]  # 订单编号
            order_time = orders[order_id][2]  # 订单时间
            deadline = orders[order_id][3]  # 截止时间
            if current_time > deadline:  # 如果当前时间超过截止时间
                penalty += 1000  # 添加惩罚项，可根据具体情况调整
    # print(total_distance, penalty)
    return 1 / (total_distance + penalty + 0.001)  # 加入惩罚项的适应度计算



# 选择操作
def selection(population, fitnesses):
    selected_indices = np.random.choice(range(len(population)), size=2, p=fitnesses / fitnesses.sum(), replace=False)
    return population[selected_indices[0]], population[selected_indices[1]]


# 交叉操作
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        children1 = []
        children2 = []
        for i in range(min(len(parent1), len(parent2))):  # 修改此处
            if len(parent1[i]) > 2 and len(parent2[i]) > 2:
                point = random.randint(1, min(len(parent1[i]), len(parent2[i])) - 2)
                child1 = parent1[i][:point] + [p for p in parent2[i] if p not in parent1[i][:point]]
                child2 = parent2[i][:point] + [p for p in parent1[i] if p not in parent2[i][:point]]
            else:
                child1 = parent1[i]
                child2 = parent2[i]
            if child1:  # 检查孩子列表中是否有空列表
                children1.append(child1)
            if child2:  # 检查孩子列表中是否有空列表
                children2.append(child2)

        for child in children1:
            if child[-1][0] != child[0][0]:
                child.append(child[0])
        for child in children2:
            if child[-1][0] != child[0][0]:
                child.append(child[0])

        return children1, children2
    else:
        return parent1, parent2


def pmx_crossover(parent1, parent2):
    def pmx_single(parent_a, parent_b):
        size = len(parent_a)
        child = [None] * size

        # 选择两个交叉点
        point1, point2 = sorted(random.sample(range(1, size - 1), 2))

        # 将交叉点之间的部分复制到子代
        child[point1:point2] = parent_a[point1:point2]

        # 处理交叉点之外的部分
        for i in range(point1, point2):
            if parent_b[i] not in child:
                j = i
                while point1 <= j < point2:
                    j = parent_b.index(parent_a[j])
                child[j] = parent_b[i]

        # 处理剩余位置
        for i in range(size):
            if child[i] is None:
                child[i] = parent_b[i]

        return child

    if random.random() < crossover_rate:
        children1 = []
        children2 = []

        for i in range(min(len(parent1), len(parent2))):
            parent1_nodes = set([node for node in parent1[i] if node[0] != parent1[i][0][0]])
            parent2_nodes = set([node for node in parent2[i] if node[0] != parent2[i][0][0]])

            if parent1_nodes == parent2_nodes:  # 只有节点集相同时才执行交叉
                if len(parent1[i]) > 3 and len(parent2[i]) > 3:
                    child1 = pmx_single(parent1[i], parent2[i])
                    child2 = pmx_single(parent2[i], parent1[i])
                else:
                    child1 = parent1[i]
                    child2 = parent2[i]
                if child1:
                    children1.append(child1)
                if child2:
                    children2.append(child2)
            else:
                children1.append(parent1[i])
                children2.append(parent2[i])

        # 确保所有路径以配送中心结束
        for child in children1:
            if child[-1][0] != child[0][0]:
                child.append(child[0])
        for child in children2:
            if child[-1][0] != child[0][0]:
                child.append(child[0])

        return children1, children2
    else:
        return parent1, parent2


def individual_fix(individual):
    # 移除超出最大飞行距离的部分
    for path in individual:
        if len(path) > 2:
            path_distance = sum(calculate_distance(path[i][1], path[i + 1][1]) for i in range(len(path) - 1))
            if path_distance > max_distance:
                excess_distance = path_distance - max_distance
                while excess_distance > 0 and len(path) > 2:
                    last_point = path.pop()
                    excess_distance -= calculate_distance(last_point[1], path[-1][1])

    # 检查是否有卸货点未配送
    unvisited_points = [point for point in points if point not in sum(individual, [])]
    new_path = [[] for _ in centers]  # 为每个配送中心初始化一条路径
    while unvisited_points:
        point = unvisited_points.pop(0)
        # 找到最近的配送中心
        nearest_center = min(centers, key=lambda center: calculate_distance(center[1], point[1]))
        # 获取当前路径
        path_index = centers.index(nearest_center)
        # print(path_index)
        path = new_path[path_index]
        if not path:
            path.append(nearest_center)  # 起始点为配送中心
        current_distance = sum(calculate_distance(path[i][1], path[i + 1][1]) for i in range(len(path) - 1))
        distance_to_point = calculate_distance(path[-1][1], point[1])
        # 判断加入当前点后是否超过最大距离
        if current_distance + distance_to_point + calculate_distance(point[1], nearest_center[1]) <= max_distance and len(path) < n + 1:
            path.append(point)
        else:
            path.append(nearest_center)  # 返回配送中心
            path = [nearest_center, point]  # 开始新路径
        new_path[path_index] = path

    for path in new_path:
        if path:
            individual.append(path)
    # 确保所有路径以配送中心结束
    for path in individual:
        if path and path[-1][0] != path[0][0]:
            path.append(path[0])
    unvisited_points = [point for point in points if point not in sum(individual, [])]
    if unvisited_points:
        individual_fix(individual)
    return individual


def mutate(individual):
    for path in individual:
        if random.random() < mutation_rate and len(path) > 2:
            mutation_type = random.randint(1, 2)  # 随机选择变异类型
            # mutation_type = 2
            unloading_points = [idx for idx, point in enumerate(path) if idx != 0 and idx != len(path) - 1]

            if mutation_type == 1 and len(unloading_points) > 1:  # 路径交换，确保至少有两个卸货点
                swap_points = random.sample(unloading_points, 2)
                path[swap_points[0]], path[swap_points[1]] = path[swap_points[1]], path[swap_points[0]]
            elif mutation_type == 2 and len(unloading_points) > 1:  # 路径倒置，确保至少有两个卸货点
                reverse_start = random.choice(unloading_points)
                valid_end_points = [idx for idx in unloading_points if idx > reverse_start]
                if valid_end_points:  # 确保有合法的结束点
                    reverse_end = random.choice(valid_end_points)
                    path[reverse_start:reverse_end + 1] = reversed(path[reverse_start:reverse_end + 1])

    # 使用 individual_fix 函数修复路径
    individual = individual_fix(individual)
    return individual

def genetic_algorithm(centers, points, population_size, generations, map_size, max_distance, orders):
    population = initialize_population(centers, points, population_size, max_distance, n)
    best_individual = None
    best_fitness = float('-inf')

    current_time = 0  # 获取当前时间

    for generation in range(generations):
        fitnesses = np.array([fitness(ind, orders, current_time) for ind in population])  # 传入订单信息和当前时间

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = pmx_crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])

        population = new_population

        current_best = population[np.argmax(fitnesses)]
        current_fitness = max(fitnesses)

        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_individual = current_best
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return best_individual


# 绘制地图
def plot_map(centers, points, paths):
    plt.figure(figsize=(12, 12))

    # 绘制配送中心
    centers_x, centers_y = zip(*[center[1] for center in centers])
    centers_labels = [center[0] for center in centers]
    plt.scatter(centers_x, centers_y, c='red', marker='s', s=100, label='配送中心')
    for i, txt in enumerate(centers_labels):
        plt.annotate(txt, (centers_x[i], centers_y[i]), fontsize=12, fontweight='bold', ha='right')

    # 绘制卸货点
    points_x, points_y = zip(*[point[1] for point in points])
    points_labels = [point[0] for point in points]
    plt.scatter(points_x, points_y, c='green', marker='o', s=60, label='卸货点')
    for i, txt in enumerate(points_labels):
        plt.annotate(txt, (points_x[i], points_y[i]), fontsize=10, ha='right')

    # 绘制路径
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths)))  # 使用不同颜色绘制每条路径
    for path, color in zip(paths, colors):
        path_x, path_y = zip(*[point[1] for point in path])
        plt.plot(path_x, path_y, '-', color=color, linewidth=2, alpha=0.7, label='路径')
        for point in path:
            plt.annotate(point[0], point[1], fontsize=8, ha='center')

    plt.title('无人机配送路径规划', fontsize=16, fontweight='bold')
    plt.xlabel('X 坐标', fontsize=14)
    plt.ylabel('Y 坐标', fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()



# 生成地图
centers, points = generate_map(num_centers, num_points, map_size, max_distance)

# 生成确定性地图
# centers, points = generate_deterministic_map()

# 生成订单
orders = generate_orders(points)

# 运行遗传算法
best_individual = genetic_algorithm(centers, points, population_size, generations, map_size, max_distance, orders)

# 输出最优路径和路径长度
print("最优路径和路径长度：")
for path in best_individual:
    path_str = " -> ".join(str(point[0]) for point in path)
    path_distance = sum(calculate_distance(path[i][1], path[i + 1][1]) for i in range(len(path) - 1))
    print(f"路径: {path_str}，长度: {path_distance:.2f}")

# 绘制最佳路径
plot_map(centers, points, best_individual)
