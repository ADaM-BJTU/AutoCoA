import re
import string

def normalize_answer(s):
    """Enhanced version of answer normalization processing"""

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
        # Remove various quotes and special markings
        return re.sub(r'[""\'\'「」『』\(\)\[\]\{\}]', '', text)
    
    # Add functionality to remove common filler words
    def remove_fillers(text):
        fillers = ['well', 'so', 'basically', 'actually', 'literally', 'simply', 'just', 'um', 'uh']
        pattern = r'\b(' + '|'.join(fillers) + r')\b'
        return re.sub(pattern, ' ', text)
    
    # Normalize number representations
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
    """Enhanced version for answer extraction"""
    assert method in ['strict', 'flexible', 'comprehensive']
    
    solution_str = solution_str.strip()
    final_answer = None
    
    # First handle the </think> tag, removing the thinking process before it
    solution_str = solution_str.split('</think>')[-1].strip()
    
    if method == 'strict':
        # Strict mode only accepts answers in \boxed{} format
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
    
    elif method == 'flexible':
        # Flexible mode first tries common markers, then other patterns
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
        # Comprehensive mode tries multiple extraction strategies and selects the most likely answer
        candidates = []
        
        # 1. Check for \boxed{} format
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            candidates.append(normalize_answer(boxes[-1].strip()))
        
        # 2. Check for direct answer declarations
        patterns = [
            r"(The answer is|Therefore|Thus|So|In conclusion|Hence)[:\s]+([^\.]+)",
            r"(I believe the answer is|The final answer is|The correct answer is)[:\s]+([^\.]+)",
            r"(Answer)[:\s]+([^\.]+)"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, solution_str, re.IGNORECASE)
            for match in matches:
                candidates.append(normalize_answer(match.group(2).strip()))
        
        # 3. Check the final sentence as an answer
        if solution_str:
            sentences = [s.strip() for s in solution_str.split('.') if s.strip()]
            if sentences:
                # 添加最后一句和倒数第二句作为候选
                if len(sentences) > 0:
                    candidates.append(normalize_answer(sentences[-1]))
                if len(sentences) > 1:
                    candidates.append(normalize_answer(sentences[-2]))
        
        # Choose the most likely candidate from the options (simply selecting the first non-empty answer here)
        for candidate in candidates:
            if candidate:
                final_answer = candidate
                break
    
    return final_answer


def default_compute_score(solution_str, ground_truth, method='flexible', format_score=0.00, score=1.0):
    """增强版评分函数
    
    Args:
        solution_str: 模型的解答文本
        ground_truth: 标准答案
        method: 答案提取方法，可选 'strict'、'flexible' 或 'comprehensive'
        format_score: 格式正确但答案不完全匹配时的分数
        score: 完全匹配的满分
    """
    # Preprocessing check
    if not solution_str or not ground_truth:
        return 0.0
    
    # Format check (if checking for </think> format is needed)
    format_correct = '</think>' in solution_str 
    if not format_correct:
        return 0.0
    elif '<begin_search>' in solution_str.split('</think>')[-1]:
        return 0.0
    
    # Extract and normalize answers
    answer = extract_solution(solution_str=solution_str, method=method)
    normalized_ground_truth = normalize_answer(ground_truth)
    

    if answer is None:
        return 0.0
    
    if answer == normalized_ground_truth:
        return score
    
    if normalized_ground_truth in answer or answer in normalized_ground_truth:
        return format_score
    
    gt_words = set(normalized_ground_truth.split())
    answer_words = set(answer.split())
    common_words = gt_words.intersection(answer_words)
    
    if len(common_words) >= len(gt_words) * 0.7:
        return score
    
    return 0.0


def compute_score(solution_str, ground_truth, method='flexible', format_score=0.00, score=1.0):
    # Multiple answers are separated by "<ans_split>"
    answers = ground_truth.split("<ans_split>")
    
    for ans in answers:
        if default_compute_score(solution_str,ans, method='flexible', format_score=0.00, score=1.0) >=0.98:
            return 1.
    
            
    return 0.


        