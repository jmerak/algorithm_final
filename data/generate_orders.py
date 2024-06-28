import random


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
