import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable

class GeneticCalibrator:
    def __init__(self,
                 population_size: int,
                 action_dim: int,
                 fitness_func: Callable,
                 target_state: np.ndarray,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_ratio: float = 0.1):
        """
        遗传算法校准器
        :param population_size: 种群规模
        :param action_dim: 动作维度
        :param fitness_func: 适应度函数 (action -> fitness_score)
        :param target_state: 目标系统状态
        :param crossover_rate: 交叉概率 (0.6~0.9)
        :param mutation_rate: 变异概率 (0.01~0.1)
        :param elite_ratio: 精英保留比例
        """
        self.pop_size = population_size
        self.action_dim = action_dim
        self.fitness_func = fitness_func
        self.target = target_state
        self.pc = crossover_rate
        self.pm = mutation_rate
        self.elite_num = int(population_size * elite_ratio)
        # 种群初始化时确保浮点类型
        self.population = np.random.uniform(
            -1, 1,
            (population_size, action_dim)
        ).astype(np.float64)
        # # 初始化种群 [-1, 1]区间随机动作
        # self.population = np.random.uniform(-1, 1, (population_size, action_dim))

    def evaluate_fitness(self) -> np.ndarray:
        """评估种群适应度"""
        return np.array([self.fitness_func(ind) for ind in self.population])

    # def select_parents(self, fitness: np.ndarray) -> List[np.ndarray]:
    #     """轮盘赌选择父代"""
    #     probs = fitness / fitness.sum()
    #     parent_indices = np.random.choice(
    #         self.pop_size,
    #         size=self.pop_size - self.elite_num,
    #         p=probs
    #     )
    #     return self.population[parent_indices]
    def select_parents(self, fitness: np.ndarray) -> List[np.ndarray]:
        """增强版轮盘赌选择"""
        candidate_indices = np.arange(self.pop_size)

        # 处理全零适应度的极端情况
        if np.all(fitness <= 1e-6):
            probs = np.ones(self.pop_size) / self.pop_size
        else:
            probs = fitness / (fitness.sum() + 1e-8)  # 防止除零

        probs = np.clip(probs, 1e-6, 1.0)  # 限制概率范围
        probs /= probs.sum()  # 重新归一化

        # 使用确定性的索引选择
        parent_indices = np.random.choice(
            candidate_indices,
            size=max(2, self.pop_size - self.elite_num),  # 保证至少选择2个父代
            p=probs,
            replace=True
        )
        return self.population[parent_indices]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """单点交叉"""
        if np.random.rand() < self.pc:
            pt = np.random.randint(1, self.action_dim)
            child = np.concatenate([parent1[:pt], parent2[pt:]])
            return child
        return parent1.copy()

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """高斯变异"""
        mask = np.random.rand(self.action_dim) < self.pm
        noise = np.random.normal(0, 0.1, self.action_dim)
        return individual + mask * noise

    # def evolve(self):
    #     """执行一代进化"""
    #     fitness = self.evaluate_fitness()
    #
    #     # 精英保留
    #     elite_indices = fitness.argsort()[-self.elite_num:]
    #     elites = self.population[elite_indices]
    #
    #     # 选择父代
    #     parents = self.select_parents(fitness)
    #
    #     # 生成子代
    #     children = []
    #     for i in range(0, len(parents) - 1, 2):
    #         child1 = self.crossover(parents[i], parents[i + 1])
    #         child2 = self.crossover(parents[i + 1], parents[i])
    #         children.extend([self.mutate(child1), self.mutate(child2)])
    #
    #     # 新种群 = 精英 + 子代
    #     self.population = np.concatenate([elites, np.array(children)[:self.pop_size - self.elite_num]])
    # def evolve(self):
    #     """带安全检查的进化流程"""
    #     # 检查种群完整性
    #     if len(self.population) != self.pop_size:
    #         self.population = self.population[:self.pop_size]
    #
    #     fitness = self.evaluate_fitness()
    #
    #     # 精英选择
    #     elite_indices = np.argpartition(fitness, -self.elite_num)[-self.elite_num:]
    #     elites = self.population[elite_indices]
    #
    #     # 父代选择
    #     parents = self.select_parents(fitness)
    #
    #     # 交叉和变异
    #     children = []
    #     for i in range(0, len(parents), 2):
    #         if i + 1 >= len(parents):
    #             break
    #         child1 = self.crossover(parents[i], parents[i + 1])
    #         child2 = self.crossover(parents[i + 1], parents[i])
    #         children.append(self.mutate(child1))
    #         children.append(self.mutate(child2))
    #
    #     # 种群重组
    #     new_pop = np.vstack([elites, np.array(children)])
    #     self.population = new_pop[:self.pop_size]  # 严格保持种群大小
    def evolve(self):
        """安全增强版进化流程"""
        fitness = self.evaluate_fitness()

        # 精英选择（带边界检查）
        sorted_indices = np.argsort(fitness)
        elite_indices = sorted_indices[-self.elite_num:]
        elite_indices = np.clip(elite_indices, 0, self.pop_size - 1)
        elites = self.population[elite_indices]

        # 父代选择（带尺寸验证）
        parents = self.select_parents(fitness)
        assert len(parents) >= 2, "至少需要两个父代进行繁殖"

        # 子代生成（处理奇数情况）
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                child1 = self.crossover(parents[i], parents[i + 1])
                child2 = self.crossover(parents[i + 1], parents[i])
                children += [self.mutate(child1), self.mutate(child2)]
            else:
                # 单亲繁殖：克隆并变异
                child = self.mutate(parents[i].copy())
                children.append(child)

            # 提前终止避免溢出
            if len(children) >= (self.pop_size - self.elite_num):
                break

        # 种群重组（强制尺寸一致）
        new_pop = np.vstack([elites, children[:self.pop_size - self.elite_num]])

        # 最终检查
        if len(new_pop) != self.pop_size:
            diff = self.pop_size - len(new_pop)
            new_pop = np.vstack([new_pop, self.population[:diff]])

        self.population = new_pop[:self.pop_size]

    def get_best_action(self) -> np.ndarray:
        """获取当前最优动作"""
        fitness = self.evaluate_fitness()
        return self.population[np.argmax(fitness)]


# 示例系统模型
class VirtualSystem:
    def __init__(self, true_action: np.ndarray):
        """模拟需要校准的真实系统（动作到状态的映射）"""
        self.true_action = true_action  # 假设的真实最优动作

    def get_state(self, action: np.ndarray) -> np.ndarray:
        """获取系统状态（含噪声）"""
        error = np.linalg.norm(action - self.true_action)
        noise = np.random.normal(0, 0.1, len(self.true_action))
        return np.array([1.0 / (1 + error)]) + noise


# 使用示例
if __name__ == "__main__":
    # 系统参数
    TARGET_STATE = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 目标状态
    TRUE_ACTION = np.array([0., -0., -0., -0., -0.])  # 假设的真实最优动作
    ACTION_DIM = 5  # 动作维度

    # 初始化虚拟系统
    system = VirtualSystem(TRUE_ACTION)


    # 定义适应度函数
    def fitness_function(action):
        current_state = system.get_state(action)
        return -np.abs(current_state - TARGET_STATE).sum() / len(current_state)


    # 初始化遗传算法
    ga = GeneticCalibrator(
        population_size=100,
        action_dim=ACTION_DIM,
        fitness_func=fitness_function,
        target_state=TARGET_STATE,
        crossover_rate=0.85,
        mutation_rate=0.05
    )
    # 安全参数组合
    SAFE_PARAMS = {
        'population_size': 300,  # 推荐100以上确保多样性
        'crossover_rate': 0.85,  # 较高交叉率促进信息交换
        'mutation_rate': 0.05,  # 适度变异维持多样性
        'elite_ratio': 0.1,  # 保留前10%精英
        'mutation_scale': 0.1,  # 高斯变异标准差
        'min_prob': 1e-6  # 最小选择概率
    }

    # 进化循环
    MAX_GENERATIONS = 10000
    TOLERANCE = 0.05

    # for gen in range(MAX_GENERATIONS):
    #     ga.evolve()
    #     best_action = ga.get_best_action()
    #     current_state = system.get_state(best_action)
    #
    #     print(f"Gen {gen + 1}: Best state {current_state[0]:.4f} with action {np.round(best_action, 4)}")
    #
    #     if np.abs(current_state - TARGET_STATE) < TOLERANCE:
    #         print(f"\nSuccess in {gen + 1} generations!")
    #         print(f"Optimal action: {best_action}")
    #         break
    best_action = ga.get_best_action()
    current_state = system.get_state(best_action)
    for i in range(best_action.size):
        locals()['best_action_' + str(i)] = [best_action[i]]  # 状态向量化，添加新的状态便于绘制训练曲线
    # 修改进化循环中的判断条件
    for gen in range(MAX_GENERATIONS):
        ga.evolve()
        best_action = ga.get_best_action()
        current_state = system.get_state(best_action)
        # 增加3
        for i in range(best_action.size):
            locals()['best_action_' + str(i)].append(best_action[i])  # 状态向量化，添加新的状态便于绘制训练曲线

        print(f"Gen {gen + 1}: Best state {current_state[0]:.4f} with action {np.round(best_action, 4)}")

        # 修改后的判断条件
        if (np.abs(current_state - TARGET_STATE) < TOLERANCE).all():
            print(f"\nSuccess in {gen + 1} generations!")
            print(f"Optimal action: {best_action}")
            break

# 可视化
    for i in range(best_action.size):
        plt.plot(best_action[i])
    plt.title("Convergence Process")
    plt.xlabel("Iterations")
    plt.ylabel("best_action")
    plt.show()


# 扩展
# 增强选择机制
# def tournament_selection(self, fitness, k=3):
#     """锦标赛选择"""
#     selected = []
#     for _ in range(self.pop_size):
#         candidates = np.random.choice(self.pop_size, k)
#         winner = candidates[np.argmax(fitness[candidates])]
#         selected.append(self.population[winner])
#     return np.array(selected)

# 自适应参数调整
# def adaptive_parameters(self, gen):
#     """根据进化代数自动调整参数"""
#     self.pc = 0.9 - 0.5 * (gen/MAX_GENERATIONS)
#     self.pm = 0.1 + 0.15 * (gen/MAX_GENERATIONS)

# 多目标优化
# def pareto_front(self):
#     """帕累托前沿计算"""
#     # 实现多目标优化逻辑
