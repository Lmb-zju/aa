import math
import heapq


def beam_search(start_token, beam_width, max_len, model):
    """
    Beam Search 算法实现
    :param start_token: 起始 token (字符串)
    :param beam_width: beam宽度（保留的候选数量）
    :param max_len: 生成序列的最大长度
    :param model: 模拟的模型预测函数（返回概率分布）
    """

    # 初始化候选序列列表，每个元素是 (累计对数概率, 当前序列, 是否结束)
    candidates = [(0.0, [start_token], False)]

    for step in range(max_len):
        new_candidates = []

        # 如果所有候选序列都已结束则提前终止
        if all([c[2] for c in candidates]):
            break

        for log_prob, sequence, finished in candidates:
            if finished:
                # 已结束的序列直接加入新候选
                heapq.heappush(new_candidates, (log_prob, sequence, True))
                continue

            # 获取模型预测的下一个 token 概率分布（这里使用伪模型）
            last_token = sequence[-1]
            next_probs = model(last_token)  # 返回字典 {token: prob}

            # 扩展所有可能的后续 token
            for token, prob in next_probs.items():
                new_log_prob = log_prob + math.log(prob)
                new_seq = sequence + [token]

                # 判断是否结束（假设以 <END> 作为结束符）
                is_finished = (token == "<END>") or (len(new_seq) >= max_len)

                # 加入临时候选列表
                heapq.heappush(new_candidates, (new_log_prob, new_seq, is_finished))

        # 保留概率最高的前 beam_width 个候选
        # 使用堆结构优化排序效率
        candidates = heapq.nlargest(beam_width, new_candidates, key=lambda x: x[0])

    # 返回最终候选列表（按概率从高到低排序）
    return sorted(candidates, key=lambda x: -x[0])


# 示例使用的伪模型（实际应该替换为真实模型）
def pseudo_model(token):
    """模拟模型预测下一个 token 的概率分布"""
    if token == "<usr>":
        return {"A1": 0.6, "A2": 0.2, "A3": 0.2}  # 60%概率选A，40%选B
    elif token in ['A1', 'A2', 'A3']:
        return {"B1": 0.5, "B2": 0.2, "B3": 0.3,}
    elif token in ['B1', 'B2', 'B3']:
        return {"C1": 0.1, "C2": 0.4, "C3": 0.5,}
    else:
        return {"<item>": 1.0}


# 运行示例
if __name__ == "__main__":
    results = beam_search(
        start_token="<usr>",
        beam_width=2,
        max_len=5,
        model=pseudo_model
    )

    print("最终候选序列：")
    for log_prob, seq, finished in results:
        print(f"序列: {' → '.join(seq)}, 对数概率: {log_prob:.3f}")
