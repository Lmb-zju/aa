import numpy as np


class OpticalState:
    def __init__(self, output_differences):
        """
        光学状态的专业表示

        :param output_differences: 相邻输出测量值数组 (20维)
        """
        # 数值稳定性处理 (避免除零)
        min_val = np.min(output_differences)
        if min_val < 1e-3:  # 防止过小值
            min_val = 1e-3

        # 计算相对强度比例
        self.relative_intensity = output_differences / min_val - 1

        # 存储原始测量值
        self.raw_differences = output_differences.copy()

        # 计算系统均匀性指标
        self.uniformity = 1 / (1 + np.std(self.relative_intensity))

    @property
    def vector(self):
        """获取状态向量 (用于神经网络输入)"""
        return self.relative_intensity

    @property
    def range_constraint(self):
        """应用物理范围约束 [0, 8]"""
        return np.clip(self.relative_intensity, 0, 8)

    def is_valid(self):
        """验证状态是否物理可实现"""
        # 检查1: 所有差分值应为正
        if np.any(self.raw_differences <= 0):
            return False

        # 检查2: 相对强度在合理范围
        if np.any(self.relative_intensity < 0) or np.any(self.relative_intensity > 10):
            return False

        # 检查3: 单调性 (输出应递增)
        if not np.all(np.diff(self.raw_differences) >= 0):
            return False

        return True


# 使用示例
# output_diff = np.array([0.1 + i*0.01 for i in range(19)])  # 20个测量值
# output_diff = np.array([0.1] * 19)  # 或者用具体的19个值
output_diff = np.random.normal(8, 4, [1, 19])
state = OpticalState(output_diff)

print("状态向量:", state.vector)
print("约束后状态:", state.range_constraint)
print("状态有效性:", state.is_valid())
print("系统均匀性:", state.uniformity)
