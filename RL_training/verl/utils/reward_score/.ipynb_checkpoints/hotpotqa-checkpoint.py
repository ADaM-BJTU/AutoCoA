import re
import string

def normalize_answer(s):
    """增强版答案归一化处理"""
    if s is None:
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    def remove_special_tokens(text):
        # 移除各种引号和特殊标记
        return re.sub(r'[""\'\'「」『』\(\)\[\]\{\}]', '', text)
    
    # 添加移除常见无关词的功能
    def remove_fillers(text):
        fillers = ['well', 'so', 'basically', 'actually', 'literally', 'simply', 'just', 'um', 'uh']
        pattern = r'\b(' + '|'.join(fillers) + r')\b'
        return re.sub(pattern, ' ', text)
    
    # 规范化数字表示
    def normalize_numbers(text):
        text = re.sub(r'\bzero\b', '0', text)
        text = re.sub(r'\bone\b', '1', text)
        text = re.sub(r'\btwo\b', '2', text)
        text = re.sub(r'\bthree\b', '3', text)
        text = re.sub(r'\bfour\b', '4', text)
        text = re.sub(r'\bfive\b', '5', text)
        text = re.sub(r'\bsix\b', '6', text)
        text = re.sub(r'\bseven\b', '7', text)
        text = re.sub(r'\beight\b', '8', text)
        text = re.sub(r'\bnine\b', '9', text)
        return text
    
    # 执行所有归一化步骤
    return white_space_fix(remove_fillers(normalize_numbers(
        remove_special_tokens(remove_articles(remove_punc(lower(s)))))))

def extract_solution(solution_str, method='comprehensive'):
    """提取答案的增强版本"""
    assert method in ['strict', 'flexible', 'comprehensive']
    
    solution_str = solution_str.strip()
    final_answer = None
    
    # 首先处理 </think> 标记，将其之前的思考过程剔除
    solution_str = solution_str.split('</think>')[-1].strip()
    
    if method == 'strict':
        # 严格模式只接受 \boxed{} 格式的答案
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
    
    elif method == 'flexible':
        # 灵活模式先尝试常见标记，然后尝试其他模式
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
        else:
            # 查找常见答案前缀
            answer_pattern = re.search(r"(The answer is|Therefore,|Thus,|So,|In conclusion,|Hence,)[:\s]+([^\.]+)", solution_str, re.IGNORECASE)
            if answer_pattern:
                final_answer = normalize_answer(answer_pattern.group(2).strip())
            elif solution_str:
                sentences = solution_str.split('.')
                if sentences:
                    final_answer = normalize_answer(sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip())
    
    elif method == 'comprehensive':
        # 综合模式尝试多种提取策略并选择最可能的答案
        candidates = []
        
        # 1. 检查 \boxed{} 格式
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            candidates.append(normalize_answer(boxes[-1].strip()))
        
        # 2. 检查直接答案声明
        patterns = [
            r"(The answer is|Therefore|Thus|So|In conclusion|Hence)[:\s]+([^\.]+)",
            r"(I believe the answer is|The final answer is|The correct answer is)[:\s]+([^\.]+)",
            r"(Answer)[:\s]+([^\.]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, solution_str, re.IGNORECASE)
            for match in matches:
                candidates.append(normalize_answer(match.group(2).strip()))
        
        # 3. 检查末尾句子作为答案
        if solution_str:
            sentences = [s.strip() for s in solution_str.split('.') if s.strip()]
            if sentences:
                # 添加最后一句和倒数第二句作为候选
                if len(sentences) > 0:
                    candidates.append(normalize_answer(sentences[-1]))
                if len(sentences) > 1:
                    candidates.append(normalize_answer(sentences[-2]))
        
        # 选择候选答案中最可能的一个（这里简单地选择第一个非空答案）
        for candidate in candidates:
            if candidate:
                final_answer = candidate
                break
    
    return final_answer

# def check_format(solution_str):
#     """检查解答格式是否符合要求"""
#     return '</think>' in solution_str

def default_compute_score(solution_str, ground_truth, method='flexible', format_score=0.00, score=1.0):
    """增强版评分函数
    
    Args:
        solution_str: 模型的解答文本
        ground_truth: 标准答案
        method: 答案提取方法，可选 'strict'、'flexible' 或 'comprehensive'
        format_score: 格式正确但答案不完全匹配时的分数
        score: 完全匹配的满分
    """
    # 预处理检查
    if not solution_str or not ground_truth:
        return 0.0
    
    # 格式检查（如果需要检查 </think> 格式）
    format_correct = '</think>' in solution_str 
    if not format_correct:
        return 0.0
    elif '<begin_search>' in solution_str.split('</think>')[-1]:
        return 0.0
    
    # 提取和标准化答案
    answer = extract_solution(solution_str=solution_str, method=method)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # 输出调试信息
    # print(f"Extracted: '{answer}', Ground Truth: '{normalized_ground_truth}'")
    
    # 评分逻辑
    if answer is None:
        return 0.0
    
    # 完全匹配
    if answer == normalized_ground_truth:
        return score
    
    # 部分匹配（如果标准答案包含在提取的答案中）
    if normalized_ground_truth in answer or answer in normalized_ground_truth:
        return format_score
    
    # 检查关键词匹配
    gt_words = set(normalized_ground_truth.split())
    answer_words = set(answer.split())
    common_words = gt_words.intersection(answer_words)
    
    # 如果有超过60%的关键词匹配，给部分分数
    if len(common_words) >= len(gt_words) * 0.7:
        return score
    
    return 0.0


def compute_score(solution_str, ground_truth, method='flexible', format_score=0.00, score=1.0):
    """增强版评分函数
    
    Args:
        solution_str: 模型的解答文本
        ground_truth: 标准答案
        method: 答案提取方法，可选 'strict'、'flexible' 或 'comprehensive'
        format_score: 格式正确但答案不完全匹配时的分数
        score: 完全匹配的满分
    """
    answers = ground_truth.split("<ans_split>")
    
    for ans in answers:
        if default_compute_score(solution_str,ans, method='flexible', format_score=0.00, score=1.0) >=0.98:
            return 1.
    
            
    return 0.


        