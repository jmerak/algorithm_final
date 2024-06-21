import numpy as np
import random
import matplotlib.pyplot as plt
from pylab import mpl
import datetime  # 新增导入datetime模块


# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 参数
num_centers = 5
num_points = 10
map_size = 10
t = 30
n = 5
max_distance = 20
speed = 60
time_limit = 24 * 60 // t
priority_weights = {'一般': 1, '较紧急': 2, '紧急': 3}
population_size = 5000
generations = 10
mutation_rate = 0.1
crossover_rate = 0.8

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

# 修复生成订单函数，确保每个订单元组包含订单时间和截止时间
def generate_orders(points):
    orders = []
    current_time = datetime.datetime.now()  # 获取当前时间
    for point in points:
        num_orders = random.randint(1, 3)  # 每个卸货点都至少有一个订单
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            order_time = current_time
            if priority == '一般':
                deadline = current_time + datetime.timedelta(hours=3)
            elif priority == '较紧急':
                deadline = current_time + datetime.timedelta(hours=1.5)
            else:  # '紧急'
                deadline = current_time + datetime.timedelta(minutes=30)
            orders.append((point, priority, order_time, deadline))  # 添加订单信息
    return orders


# 修改路径规划函数：动态更新路径规划
def initialize_population(centers, points, population_size, max_distance, n):
    population = []  # 存储所有个体
    for _ in range(population_size):
        individual = []
        remaining_points = points.copy()
        for center in centers:
            path = [center]  # 起始点为配送中心
            current_distance = 0  # 当前距离
            current_load = 0  # 当前负载
            while current_load < n and remaining_points:  # 只要负载小于阈值且还有卸货点
                nearest_point = min(remaining_points, key=lambda x: calculate_distance(path[-1][1], x[1]))
                distance_to_point = calculate_distance(path[-1][1], nearest_point[1])
                if current_distance + distance_to_point + calculate_distance(nearest_point[1], center[1]) > max_distance:
                    break  # 超出距离限制，结束当前路径
                path.append(nearest_point)
                current_distance += distance_to_point
                current_load += 1
                remaining_points.remove(nearest_point)
            path.append(center)  # 最后返回配送中心
            individual.append(path)  # 添加最后一条路径到个体编码中
        population.append(individual)
    return population


# 修改 fitness 函数，确保正确访问订单元组的时间信息并计算时效性惩罚
def fitness(individual, orders, current_time):
    total_distance = 0
    total_time = 0
    penalty = 0  # 添加 penalty 变量
    for path in individual:
        path_distance = 0
        path_time = 0
        for i in range(len(path) - 1):
            path_distance += calculate_distance(path[i][1], path[i + 1][1])
            path_time += path_distance / speed  # 根据距离计算时间
        total_distance += path_distance
        total_time += path_time

    # 计算订单时效性惩罚
    for path in individual:
        for order in path[1:-1]:  # 跳过起始和终点
            order_id = order[0]  # 订单编号
            order_time = orders[order_id][2]  # 订单时间
            deadline = orders[order_id][3]  # 截止时间
            if current_time > deadline:  # 如果当前时间超过截止时间
                penalty += 1000  # 添加惩罚项，可根据具体情况调整

    return 1 / (total_distance + penalty)  # 加入惩罚项的适应度计算


# 选择操作
def selection(population, fitnesses):
    selected_indices = np.random.choice(range(len(population)), size=2, p=fitnesses / fitnesses.sum(), replace=False)
    return population[selected_indices[0]], population[selected_indices[1]]


def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        children1 = []
        children2 = []
        for i in range(len(parent1)):
            if len(parent1[i]) > 2 and len(parent2[i]) > 2:
                point = random.randint(1, min(len(parent1[i]), len(parent2[i])) - 2)
                child1 = parent1[i][:point] + [p for p in parent2[i] if p not in parent1[i][:point]]
                child2 = parent2[i][:point] + [p for p in parent1[i] if p not in parent2[i][:point]]
            else:
                child1 = parent1[i]
                child2 = parent2[i]
            children1.append(child1)
            children2.append(child2)

        # 修复路径终点为配送中心
        for child in children1:
            if child[-1][0] != child[0][0]:  # 判断是否为配送中心
                child.append(child[0])  # 添加终点配送中心
        for child in children2:
            if child[-1][0] != child[0][0]:  # 判断是否为配送中心
                child.append(child[0])  # 添加终点配送中心

        return children1, children2
    else:
        return parent1, parent2


# 变异操作
def mutate(individual):
    for path in individual:
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                if random.random() < mutation_rate:
                    j = random.randint(1, len(path) - 2)
                    path[i], path[j] = path[j], path[i]


# 修改遗传算法函数，传入订单信息
def genetic_algorithm(centers, points, population_size, generations, map_size, max_distance, orders):
    population = initialize_population(centers, points, population_size, max_distance, n)
    best_individual = None
    best_fitness = float('-inf')

    current_time = datetime.datetime.now()  # 获取当前时间

    for generation in range(generations):
        fitnesses = np.array([fitness(ind, orders, current_time) for ind in population])  # 传入订单信息和当前时间

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
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
    plt.figure(figsize=(10, 10))

    # 绘制配送中心
    centers_x, centers_y = zip(*[center[1] for center in centers])
    centers_labels = [center[0] for center in centers]
    plt.scatter(centers_x, centers_y, c='blue', marker='s', label='配送中心')
    for i, txt in enumerate(centers_labels):
        plt.annotate(txt, (centers_x[i], centers_y[i]))

    # 绘制卸货点
    points_x, points_y = zip(*[point[1] for point in points])
    points_labels = [point[0] for point in points]
    plt.scatter(points_x, points_y, c='green', marker='o', label='卸货点')
    for i, txt in enumerate(points_labels):
        plt.annotate(txt, (points_x[i], points_y[i]))

    # 绘制路径
    for path in paths:
        path_x, path_y = zip(*[point[1] for point in path])
        plt.plot(path_x, path_y, 'r-')
        for point in path:
            plt.annotate(point[0], point[1])

    plt.title('无人机配送路径规划')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()


# 生成地图
# centers, points = generate_map(num_centers, num_points, map_size, max_distance)

# 生成确定性地图
centers, points = generate_deterministic_map()

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
