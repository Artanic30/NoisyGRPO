import os
import re
from datetime import datetime
import json
import torch
import spacy
import string
from sentence_transformers import SentenceTransformer, util
import numpy as np
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL', 'rouge2'], use_stemmer=True)
nlp = spacy.load("en_core_web_lg")


class BidirectionalEmbeddingMatching:
    def __init__(self,
                 model_name="/inspurfs/group/hexm/VL_data/pretrained_models/sentence-transformers/all-mpnet-base-v2"):
        """
        初始化 BEM 相似度计算模型
        :param model_name: 预训练 Transformer 模型名称 (支持 BERT, RoBERTa, DeBERTa 等)
        """
        if not os.path.exists(model_name):
            model_name = model_name.replace('/inspurfs/group/hexm/VL_data/pretrained_models', '/2022233227/pretrained_models')
        self.model = SentenceTransformer(model_name).cuda()

    def get_token_embeddings(self, sentence):
        """
        获取句子的 token 级别嵌入
        :param sentence: 输入文本
        :return: token 级别嵌入 (Tensor) 和句子嵌入 (Tensor)
        """
        tokens = self.model.tokenize([sentence])
        for k, v in tokens.items():
            tokens[k] = v.to(self.model.device)
        outputs = self.model(tokens, output_value="token_embeddings", convert_to_tensor=True)
        return outputs  # shape: [seq_len, hidden_dim]

    def max_pooling_similarity(self, emb1, emb2):
        """
        计算双向最大池化的 token 级别相似度
        :param emb1: 句子 1 的 token 嵌入
        :param emb2: 句子 2 的 token 嵌入
        :return: BEM 相似度分数
        """
        # 计算余弦相似度矩阵
        cosine_sim_matrix = util.pytorch_cos_sim(emb1["sentence_embedding"], emb2["sentence_embedding"])

        # 前向匹配（sentence1 → sentence2）
        forward_match = torch.max(cosine_sim_matrix, dim=1)[0]  # 每个 token 在 sentence2 上的最大相似度
        forward_score = torch.mean(forward_match)  # 取均值

        # 反向匹配（sentence2 → sentence1）
        backward_match = torch.max(cosine_sim_matrix, dim=0)[0]  # 每个 token 在 sentence1 上的最大相似度
        backward_score = torch.mean(backward_match)  # 取均值

        # BEM 最终相似度
        bem_similarity = (forward_score + backward_score) / 2
        return bem_similarity.item()

    def compute_similarity(self, sentence1, sentence2):
        """
        计算两个句子的 BEM 相似度
        :param sentence1: 句子 1
        :param sentence2: 句子 2
        :return: 相似度分数 (0~1)
        """
        emb1 = self.get_token_embeddings(sentence1)
        emb2 = self.get_token_embeddings(sentence2)
        return self.max_pooling_similarity(emb1, emb2)


bem = BidirectionalEmbeddingMatching()

def compute_rouge_reward(pred: str, ref: str) -> float:
    """
    计算 pred 与 ref 的 ROUGE 奖励分：
    reward = (ROUGE‑1 + ROUGE‑2 + ROUGE‑L) / 3  的 F1 平均值。
    """
    scores = scorer.score(ref, pred)  # 注意：先给参考答案，再给预测
    reward = (
                     scores['rouge1'].fmeasure +
                     scores['rouge2'].fmeasure +
                     scores['rougeL'].fmeasure
             ) / 3
    return reward


def zscore_scale_to_target(data, target_mean=0.55, target_variance=0.15, eps=1e-6):
    """
    使用 Z-score 标准化 + 缩放，使得数据具有指定的均值和方差
    参数：
        data: List[float] - 原始数据
        target_mean: float - 目标均值
        target_variance: float - 目标方差
    返回：
        List[float] - 缩放后的数据
    """
    data = torch.tensor(data).cpu().tolist()
    data = np.array(data, dtype=np.float32)
    orig_mean = np.mean(data)
    orig_std = np.std(data)

    if orig_std == 0:
        print("data std is 0，return original list")
        return data.tolist()

    target_std = np.sqrt(target_variance)

    # Z-score 标准化 + 缩放 + 平移
    scaled = ((data - orig_mean) / orig_std) * target_std + target_mean
    return scaled.tolist()


def multiple_choice_check(pred, gt, ques, reward):
    if 'choices:' not in ques.lower():
        return reward

    def extract_choice(text):
        # 去除所有标点符号，只保留字母和空格
        text = re.sub(r'[^\w\s]', '', text)

        # 匹配 ' A ', ' B ', ' C ', ' D ' 格式的正确选项
        match = re.search(r'\s([ABCDEFGHIJK])\s', text)
        return match.group(1) if match else None

    gt_choice = extract_choice(gt)  # 提取 Ground Truth 选项
    pred_choice = pred.strip().upper()  # 处理 pred，去空格并转换为大写

    if gt_choice is not None and pred_choice == gt_choice:
        reward = 1.0
    elif gt_choice is not None and pred_choice != gt_choice:
        reward = reward

    return reward


def contains_yes(text):
    # 使用正则表达式匹配独立的"no"单词（不区分大小写）
    pattern = r'yes\b[.,!?<]?'
    return bool(re.search(pattern, text.lower(), flags=re.IGNORECASE))


def contains_no(text):
    # 使用正则表达式匹配独立的"no"单词（不区分大小写）
    pattern = r'no\b[.,!?<]?'
    return bool(re.search(pattern, text.lower(), flags=re.IGNORECASE))


def yes_or_no_check(pred, gt, ques, reward):
    if 'Answer the question with Yes or No.' not in ques:
        return reward

    if contains_yes(gt) or contains_no(gt):
        if contains_yes(pred) and contains_yes(gt):
            return 1

        if contains_no(gt) and contains_no(pred):
            return 1

        return reward

    return reward

def debug_point():
    import torch.distributed as dist
    if dist.get_rank() == 0:
        import ipdb
        ipdb.set_trace()
    dist.barrier()

def noisy_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    num_gp = len(list(set([json.dumps(dd) for dd in kwargs['prompts']])))
    gp_size = int(len(kwargs['prompts']) / num_gp)
    rewards = [[] for _ in range(num_gp)]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>([\s\S]*?)</answer>'
    for s_idx, (content, sol, pro, n_r) in enumerate(zip(contents, solution, kwargs['prompts'], kwargs['noise_extents'])):
        reward = 1 - n_r

        gp_idx = s_idx // gp_size

        rewards[gp_idx].append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Noisy reward: {reward} -------------\n")
                f.write(f"Prompt: {pro[0]['content'][1]['text']}\n====================\n")
                f.write(f"Content: {content}\n====================\n")
                f.write(f"Solution: {sol}\n====================\n")

    cot_reward_var = kwargs['cot_reward_var']
    if isinstance(cot_reward_var, float):
        cot_reward_var = [cot_reward_var for _ in range(len(rewards))]

    rewards_list = []
    for i, var in zip(rewards, cot_reward_var):
        rewards_list.extend(zscore_scale_to_target(i, target_variance=var))

    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        with open(log_path, "a") as f:
            f.write(f"-------------  Noisy reward reweighted: {rewards_list} -------------\n")
            f.write(f"Original rewards: {rewards}\n====================\n")

    return rewards_list


def correctness_reward(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>([\s\S]*?)</answer>'
    for content, sol, pro in zip(contents, solution, kwargs['prompts']):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()

                if content_answer.strip().lower() == sol.strip().lower() or \
                        content_answer.strip().lower().replace('.', '') == sol.replace('.', '').strip().lower():
                    reward = 1.0

                reward = multiple_choice_check(content_answer.strip(), sol.strip(), pro, reward)

                reward = yes_or_no_check(content_answer.strip(), sol.strip(), pro[0]['content'][1]['text'], reward)

        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} correctness_reward: {reward} -------------\n")
                f.write(f"Prompt: {pro[0]['content'][1]['text']}\n====================\n")
                f.write(f"Content: {content}\n====================\n")
                f.write(f"Solution: {sol}\n====================\n")
    return rewards


def correctness_bem_score_reward(completions, solution, **kwargs):
    global bem
    if bem is None:
        bem = BidirectionalEmbeddingMatching()
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    answer_tag_pattern = r'<answer>([\s\S]*?)</answer>'
    for content, sol, pro in zip(contents, solution, kwargs['prompts']):
        reward = 0.0
        # Try symbolic verification first
        try:
            content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
            if content_answer_match:
                content_answer = content_answer_match.group(1).strip()

                similarity = bem.compute_similarity(content_answer, sol)

                if similarity >= 0.6:
                    reward = similarity

                reward = multiple_choice_check(content_answer.strip(), sol.strip(), pro[0]['content'][1]['text'],
                                               reward)

                reward = yes_or_no_check(content_answer.strip(), sol.strip(), pro[0]['content'][1]['text'], reward)

        except Exception as e:
            print(e)
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} correctness_bem_score_reward: {reward} -------------\n")
                f.write(f"Prompt: {pro[0]['content'][1]['text']}\n====================\n")
                f.write(f"Content: {content}\n====================\n")
                f.write(f"Solution: {sol}\n====================\n")
    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    pattern = r"<think>([\s\S]*?)</think>\s*<answer>([\s\S]*?)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



reward_funcs_registry = {
    'noisy': noisy_reward,
    "correctness": correctness_reward,
    'correctness_bem_score': correctness_bem_score_reward,
    "format": format_reward,

}

if __name__ == '__main__':
    bem = BidirectionalEmbeddingMatching()
    print(bem)
    a = bem.compute_similarity('yes', 'no')
    print(a)
