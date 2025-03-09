"""
Answer Evaluation Metrics

This module provides utilities to evaluate the correctness of model-generated answers
against gold-standard references using exact match scoring and LLM-based evaluation.
"""
import re
import string
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Dict, Any, Tuple

from tqdm import tqdm
from openai import OpenAI


def normalize_answer(text: str) -> str:
    """
    Normalize answer text by removing articles, punctuation, and extra whitespace.
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text string
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def generate_chat_completion(
    message: str, 
    temperature: float = 0.7, 
    top_p: float = 0.8, 
    max_tokens: int = 4096, 
    repetition_penalty: float = 1.05, 
    n: int = 1
) -> List[str]:
    """
    Generate completions from a language model.
    
    Args:
        message: The prompt message to send to the model
        temperature: Controls randomness (lower is more deterministic)
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        repetition_penalty: Penalty for repeating tokens
        n: Number of completions to generate
        
    Returns:
        List of generated text completions
    """
    # Set your model path here
    model = "Your model path"
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8080/v1",
    )
    
    messages = [
        {"role": "user", "content": message}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={
                "repetition_penalty": repetition_penalty,
            },
            n=n
        )
        
        texts = [choice.message.content for choice in response.choices]
        
        return texts
    except Exception as e:
        print(f"Error in chat completion: {e}")
        return []


class Metrics:
    """
    Class to calculate and track evaluation metrics for answer correctness.
    """
    
    def __init__(self, data_file_path: str = ""):
        """
        Initialize the Metrics class.
        
        Args:
            data_file_path: Path to the JSON data file containing predictions and references
        """
        self.data_file_path = data_file_path
        
    def extract_boxed_answer(self, text: str) -> str:
        """
        Extract content from LaTeX-style \\boxed{} notation.
        
        Args:
            text: Text containing boxed content
            
        Returns:
            Extracted content from the box or empty string if not found
        """
        boxed_pattern = r'\\boxed\{(.*)\}'
        boxed_match = re.search(boxed_pattern, text)
        
        if boxed_match:
            boxed_content = boxed_match.group(1)
            
            text_pattern = r'\\text\{(.*?)\}'
            
            def replace_text(match):
                return match.group(1)
            
            clean_content = re.sub(text_pattern, replace_text, boxed_content)
            return clean_content
        else:
            return ""
        
    def calculate_em(self, prediction: str, golden_answers: Union[str, List[str]]) -> float:
        """
        Calculate exact match score between prediction and golden answers.
        
        Args:
            prediction: The predicted answer
            golden_answers: One or more reference answers
            
        Returns:
            Binary score (1.0 if match, 0.0 if no match)
        """
        if isinstance(golden_answers, str):
            golden_answers = [golden_answers]
            
        score = 0.0
        prediction = normalize_answer(prediction)
        
        for golden_answer in golden_answers:
            normalized_golden = normalize_answer(golden_answer)
            if normalized_golden == prediction:
                score = 1.0
                break
                
        return score
    
    def calculate_em_metric(self, data: List[Dict[str, Any]]) -> float:
        """
        Calculate the exact match metric across all examples in the dataset.
        
        Args:
            data: List of data items containing predictions and golden answers
            
        Returns:
            Average exact match score
        """
        em_score = [0.0, 0.0, 0.0]
        
        for item in data:
            golden_answers = [normalize_answer(answer) for answer in item["golden_answers"]]
            if len(item["boxed_answer"]) != 3:
                continue
                
            for i in range(3):
                em_score[i] += self.calculate_em(item["boxed_answer"][i], golden_answers)
        
        em_score = [score / len(data) for score in em_score]
        
        print("EM Score: ", em_score)
        print("EM Average Score: ", sum(em_score) / len(em_score))
        
        return sum(em_score) / len(em_score)

    def llm_judger_parallel(self, data_file_path: str) -> None:
        """
        Use an LLM to judge answer correctness with parallel processing.
        
        Args:
            data_file_path: Path to the JSON data file to evaluate
        """
        # Load preprocessed data
        with open(data_file_path, 'r') as f:
            data = json.load(f)
            
        prompt_template = '''You are an expert evaluator assessing the correctness of a predicted answer based on a list of gold-standard answers.  

- The **gold-standard answer** are: <{gold_answers_list}>
- The **predicted answer** is: <{answer}>  

### Task:  
Determine whether the predicted answer is correct given the list of gold-standard answers. An answer is considered correct if it aligns with at least one of the gold-standard answers, even when expressed in innovative or forward-looking ways.

### Evaluation Criteria:  
1. **Exact Match**: If the predicted answer is identical to any one of the gold-standard answers in the list, it is correct.
2. **Semantics**: If the predicted answer conveys the same meaning as any one of the gold-standard answers in the list, it is correct. Innovative phrasing or forward-thinking expressions that capture the same essence are acceptable.
3. **Minor Variations**: Minor rewordings, synonyms, or slight grammatical differences that do not alter the meaning are acceptable.
4. **Incorrect Cases**: If the predicted answer omits key information, introduces incorrect details, or deviates significantly in meaning from all items in the list, it is incorrect.

### Output Format:
You should provide the final evaluation in the format of \\boxed{{YOUR_EVALUATION}} , YOUR_EVALUATION should be "Correct" or "Incorrect".'''
        
        # Define function to process a single item
        def process_item(item):
            gold_answer = item["golden_answers"]
            final_result = []
            
            for answer in item["boxed_answer"]:
                prompt_format = prompt_template.format(gold_answers_list=gold_answer, answer=answer)
                response_list = generate_chat_completion(prompt_format, n=3)
                
                result = []
                for response in response_list:
                    result.append(self.extract_boxed_answer(response))
                    
                correct_count = result.count("Correct")
                incorrect_count = result.count("Incorrect") + result.count("")
                final_result.append(correct_count > incorrect_count)
                
            item["llm_result"] = final_result
            return item
        
        # Process items in parallel using a thread pool
        max_workers = 30
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            processed_data = list(tqdm(
                executor.map(process_item, data),
                total=len(data),
                desc="Processing items"
            ))
        
        # Save the processed data
        with open(data_file_path, "w") as f:
            json.dump(processed_data, f)

    def calculate_llm_metric(self, data: List[Dict[str, Any]]) -> float:
        """
        Calculate the LLM-based evaluation metric across all examples.
        
        Args:
            data: List of data items containing LLM evaluation results
            
        Returns:
            Average LLM correctness score
        """
        llm_score = [0.0, 0.0, 0.0]
        
        for item in data:
            for i in range(len(item["llm_result"])):
                llm_score[i] += item["llm_result"][i] == True
        
        llm_num = 0
        llm_sum = 0
        
        for i in range(len(llm_score)):
            if llm_score[i] == 0:
                continue
            llm_score[i] = llm_score[i] / len(data)
            llm_sum += llm_score[i]
            llm_num += 1
        
        print("LLM Score: ", llm_score)
        print("LLM Average Score: ", llm_sum / llm_num)
        
        return llm_sum / llm_num

    def start(self) -> Tuple[float, float]:
        """
        Start the evaluation process, calculating both EM and LLM metrics.
        
        Returns:
            Tuple of (em_average_score, llm_average_score)
        """
        with open(self.data_file_path, 'r') as f:
            data = json.load(f)
            
        # print(f"=====================\n{self.data_file_path.split('/')[-1]}")
        
        self.llm_judger_parallel(self.data_file_path)
        em_ave = self.calculate_em_metric(data)
        llm_ave = self.calculate_llm_metric(data)
        
        return em_ave, llm_ave


def main():
    """
    Main function to evaluate multiple datasets.
    """
    data_folder = "Your data folder path"
    
    file_list = os.listdir(data_folder)
    
    em_sum = 0.0
    llm_sum = 0.0
    valid_files = 0
    
    for file_name in tqdm(file_list, desc="Processing files"):
            
        file_path = os.path.join(data_folder, file_name)
        metrics = Metrics(data_file_path=file_path)
        
        em_ave, llm_ave = metrics.start()
        em_sum += em_ave
        llm_sum += llm_ave
        valid_files += 1
    
    print(f"{'-'*30}")
    print(f"EM Average Score: {em_sum / valid_files:.4f}")
    print(f"LLM Average Score: {llm_sum / valid_files:.4f}")


if __name__ == "__main__":
    main()