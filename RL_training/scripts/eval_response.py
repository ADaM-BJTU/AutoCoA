import re
import string
import pandas as pd
from typing import List, Optional, Union, Dict, Any

def normalize_answer(text: Optional[str]) -> str:
    """Enhanced answer normalization process.
    
    Args:
        text: Text to be normalized, can be None
        
    Returns:
        str: Normalized text
    """
    if text is None:
        return ""
    
    # Execute all normalization steps
    return white_space_fix(remove_fillers(normalize_numbers(
        remove_special_tokens(remove_articles(remove_punc(lower(text)))))))

def remove_articles(text: str) -> str:
    """Remove articles from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with articles removed
    """
    return re.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text: str) -> str:
    """Fix whitespace issues in text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with fixed whitespace
    """
    return ' '.join(text.split())

def remove_punc(text: str) -> str:
    """Remove punctuation from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with punctuation removed
    """
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def lower(text: str) -> str:
    """Convert text to lowercase.
    
    Args:
        text: Input text
        
    Returns:
        str: Lowercase text
    """
    return text.lower()

def remove_special_tokens(text: str) -> str:
    """Remove special tokens and quotation marks from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with special tokens removed
    """
    return re.sub(r'[""\'\'「」『』\(\)\[\]\{\}]', '', text)

def remove_fillers(text: str) -> str:
    """Remove filler words from text.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with filler words removed
    """
    fillers = ['well', 'so', 'basically', 'actually', 'literally', 'simply', 'just', 'um', 'uh']
    pattern = r'\b(' + '|'.join(fillers) + r')\b'
    return re.sub(pattern, ' ', text)

def normalize_numbers(text: str) -> str:
    """Convert number words to numeric characters.
    
    Args:
        text: Input text
        
    Returns:
        str: Text with normalized numbers
    """
    number_mapping = {
        r'\bzero\b': '0', r'\bone\b': '1', r'\btwo\b': '2',
        r'\bthree\b': '3', r'\bfour\b': '4', r'\bfive\b': '5',
        r'\bsix\b': '6', r'\bseven\b': '7', r'\beight\b': '8',
        r'\bnine\b': '9'
    }
    
    for word_pattern, digit in number_mapping.items():
        text = re.sub(word_pattern, digit, text)
    return text

def extract_solution(solution_str: str, method: str = 'comprehensive') -> Optional[str]:
    """Extract the final answer from solution text.
    
    Args:
        solution_str: Text containing the solution
        method: Extraction method, options: 'strict', 'flexible', 'comprehensive'
    
    Returns:
        Optional[str]: Extracted answer, returns None if extraction fails
    """
    assert method in ['strict', 'flexible', 'comprehensive'], "Method must be 'strict', 'flexible', or 'comprehensive'"
    
    solution_str = solution_str.strip()
    final_answer = None
    
    # Remove thinking process
    solution_str = solution_str.split('</think>')[-1].strip()
    
    if method == 'strict':
        # Strict mode only accepts \boxed{} format answers
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
    
    elif method == 'flexible':
        # Flexible mode tries common markers first, then other patterns
        boxes = re.findall(r"\\boxed{([^}]*)}", solution_str)
        
        if boxes:
            final_answer = normalize_answer(boxes[-1].strip())
        else:
            # Look for common answer prefixes
            answer_pattern = re.search(r"(The answer is|Therefore,|Thus,|So,|In conclusion,|Hence,)[:\s]+([^\.]+)", 
                                      solution_str, re.IGNORECASE)
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
        
        # 3. Check last sentences as answers
        if solution_str:
            sentences = [s.strip() for s in solution_str.split('.') if s.strip()]
            if sentences:
                # Add last and second-to-last sentences as candidates
                if len(sentences) > 0:
                    candidates.append(normalize_answer(sentences[-1]))
                if len(sentences) > 1:
                    candidates.append(normalize_answer(sentences[-2]))
        
        # Select the most likely answer from candidates
        for candidate in candidates:
            if candidate:
                final_answer = candidate
                break
    
    return final_answer

def compute_score(solution_str: str, ground_truth: str, 
                  method: str = 'flexible', format_score: float = 0.00, 
                  score: float = 1.0) -> float:
    """Evaluate the score of a solution.
    
    Args:
        solution_str: Model's solution text
        ground_truth: Standard answer
        method: Answer extraction method, options: 'strict', 'flexible', 'comprehensive'
        format_score: Score when format is correct but answer doesn't fully match
        score: Full score for complete match
        
    Returns:
        float: Evaluation score
    """
    # Preprocessing checks
    if not solution_str or not ground_truth:
        return 0.0
    
    # Format check
    format_correct = '</think>' in solution_str 
    if not format_correct:
        return 0.0
    elif '<begin_search>' in solution_str.split('</think>')[-1]:
        return 0.0
    
    # Extract and normalize answers
    answer = extract_solution(solution_str=solution_str, method=method)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    # Scoring logic
    if answer is None:
        return 0.0
    
    # Complete match
    if answer == normalized_ground_truth:
        return score
    
    # Partial match
    if normalized_ground_truth in answer or answer in normalized_ground_truth:
        return format_score
    
    # Check keyword matching
    gt_words = set(normalized_ground_truth.split())
    answer_words = set(answer.split())
    common_words = gt_words.intersection(answer_words)
    
    # If over 70% of keywords match, give full score
    if len(common_words) >= len(gt_words) * 0.7:
        return score
    
    return 0.0

def evaluate_dataframe(df_path: str) -> pd.DataFrame:
    """Evaluate dataframe containing model responses.
    
    Args:
        df_path: Path to Parquet file containing model responses
        
    Returns:
        pd.DataFrame: Dataframe with evaluation results added
    """
    df = pd.read_parquet(df_path)
    
    if len(df) > 0:
        df = df.iloc[:-1]  # Remove the last row
        
    answer_list = []
    for response, answer in zip(df['responses'].tolist(), df['reward_model'].tolist()):
        flag = False
        for ans in answer['ground_truth']:
            if compute_score(response[0], ans) == 1.0:
                flag = True
                break
        answer_list.append(flag)
    
    # Add evaluation results
    df['answer'] = answer_list
    df['source_file'] = df['source_file'].apply(lambda x: x.split('/')[-1])
    
    return df

def print_evaluation_results(df: pd.DataFrame) -> None:
    """Print evaluation results.
    
    Args:
        df: Dataframe containing evaluation results
    """
    print(df.groupby('source_file')['answer'].mean())
    print(df.iloc[0].to_dict())

def main(file_path: str, output_path: str) -> None:
    """Main function.
    
    Args:
        file_path: Input data file path
        output_path: Output results file path
    """
    df = evaluate_dataframe(file_path)
    print_evaluation_results(df)
    df.to_json(output_path)

if __name__ == "__main__":
    input_file = 'stage2_3-7b_sft_123-rl-1-2.parquet'
    output_file = "sft123_rl12_pandas.json"
    main(input_file, output_file)