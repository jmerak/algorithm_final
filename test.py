import numpy as np
import random
import matplotlib.pyplot as plt


from pylab import mpl
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
population_size = 50
generations = 100
mutation_rate = 0.1
crossover_rate = 0.8


# 生成配送中心和卸货点的坐标
def generate_map(num_centers, num_points, map_size, max_distance):
    centers = [(random.uniform(0, map_size), random.uniform(0, map_size)) for _ in range(num_centers)]

    points = []
    for _ in range(num_points):
        while True:
            point = (random.uniform(0, map_size), random.uniform(0, map_size))
            if all(calculate_distance(center, point) <= max_distance / 2 for center in centers):
                points.append(point)
                break

    return centers, points


# 距离计算
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 订单生成函数
def generate_orders(points):
    orders = []
    for point in points:
        num_orders = random.randint(0, 3)
        for _ in range(num_orders):
            priority = random.choice(['一般', '较紧急', '紧急'])
            orders.append((point, priority))
    return orders


# 初始化种群
def initialize_population(centers, points, population_size, max_distance, n):
    population = []
    for _ in range(population_size):
        individual = []
        for center in centers:
            # 随机排序卸货点
            random.shuffle(points)
            path = [center]  # 起始点为配送中心
            current_distance = 0
            current_load = 0
            for point in points:
                distance_to_point = calculate_distance(path[-1], point)
                if current_distance + distance_to_point + calculate_distance(point, center) > max_distance:
                    # 当前路径已达到最大飞行距离，开始新路径
                    path.append(center)  # 返回配送中心
                    individual.append(path)  # 添加当前路径到个体编码中
                    path = [center, point]  # 新路径的起始点为配送中心和当前卸货点
                    current_distance = calculate_distance(center, point)
                    current_load = 1  # 重置当前负载
                elif current_load + 1 <= n:
                    # 当前负载未满，可以继续添加卸货点
                    path.append(point)
                    current_distance += distance_to_point
                    current_load += 1
            path.append(center)  # 最后返回配送中心
            individual.append(path)  # 添加最后一条路径到个体编码中
        population.append(individual)
    return population



# 计算适应度
def fitness(individual):
    total_distance = 0
    for path in individual:
        path_distance = 0
        for i in range(len(path) - 1):
            path_distance += calculate_distance(path[i], path[i + 1])
        total_distance += path_distance
    return 1 / total_distance


# 选择操作
def selection(population, fitnesses):
    selected_indices = np.random.choice(range(len(population)), size=2, p=fitnesses / fitnesses.sum(), replace=False)
    return population[selected_indices[0]], population[selected_indices[1]]


# 交叉操作
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, min(len(parent1[0]), len(parent2[0])) - 2)
        child1_path = parent1[0][:point] + [p for p in parent2[0] if p not in parent1[0][:point]]
        child2_path = parent2[0][:point] + [p for p in parent1[0] if p not in parent2[0][:point]]

        child1 = [child1_path]
        child2 = [child2_path]

        # Ensure child1 and child2 end with their respective centers
        child1[0].append(parent1[0][0])
        child2[0].append(parent2[0][0])

        return child1, child2
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


# 遗传算法
def genetic_algorithm(centers, points, population_size, generations, max_distance, n):
    population = initialize_population(centers, points, population_size, max_distance, n)
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitnesses = np.array([fitness(ind) for ind in population])

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = selection(population, fitnesses)
            print(parent1, parent2)
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
    centers_x, centers_y = zip(*centers)
    plt.scatter(centers_x, centers_y, c='blue', marker='s', label='配送中心')

    # 绘制卸货点
    points_x, points_y = zip(*points)
    plt.scatter(points_x, points_y, c='green', marker='o', label='卸货点')

    # 绘制路径
    for path in paths:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'r-', label='路径')

    plt.title('无人机配送路径规划')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True)
    plt.show()


# 生成地图
centers, points = generate_map(num_centers, num_points, map_size, max_distance)

# 生成订单
orders = generate_orders(points)

# 运行遗传算法
best_individual = genetic_algorithm(centers, points, population_size, generations, max_distance, n)

# 绘制最佳路径
plot_map(centers, points, best_individual)
