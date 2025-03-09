"""
Search-Augmented Reasoning System

This module implements a system for answering complex questions using a combination of
language model reasoning and web search capabilities. The system allows for multiple search
iterations to refine answers and provides structured output with explanations.
"""
import json
import sys
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add project root to path
PROJECT_PATH = str(Path(__file__).resolve().parent.parent)
sys.path.append(str(PROJECT_PATH))

# Import project modules
from models.vllm_server import generate_chat_completion, generate_completion
from models.wiki_engine import get_search_results


# Base prompt template for the reasoning assistant
REASONING_PROMPT = """
You are an intelligent reasoning assistant with access to a web search tool to help you obtain accurate information. When answering the user's questions, please adhere to the following guidelines:

1. **Uncertainty Handling:**  
   - Encourage the use of web search to solve the problems mentioned during reasoning. 
   - If you are uncertain or do not have sufficient knowledge about a specific detail, you **must** perform a web search. 
   - To perform a web search the format **must** be in the format mentioned in point 2 Web Search Format
   - You can perform multiple web searches to ensure that the answer is correct.
   - The web search should be initiated only within your internal reasoning (the "thinking" phase), and it should not appear in your final answer to the user.

2. **Web Search Format:**  
   - When calling the web search tool, use the following exact format:
     ```
     <begin_search> your query here </end_search>
     ```
   - After you perform a web search, you will receive results enclosed in:
     ```
     <search_result> ... </search_result>
     ```
   - You can then use this information to further refine your reasoning and answer.

3. **Process:**  
   - If you encounter any uncertainty or unknown knowledge, embed a web search query within your internal thoughts (surrounded by `<think></think>` tags) using the specified format.
   - Incorporate the search results into your reasoning before finalizing your answer.
   - You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}. 

Now, please answer the user's question below: 
{question}"""


def check_search_token(text: str) -> Tuple[str, str]:
    """
    Check if the text contains a search query token and extract it.
    
    Args:
        text: The text to check for search tokens
        
    Returns:
        Tuple containing the extracted search query and the text up to the end of the search token
    """
    pattern = r"<begin_search>(.*?)</end_search>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[:match.end()]
    return "", ""


def count_search_nums(text: str) -> int:
    """
    Count the number of search queries in the text.
    
    Args:
        text: The text to analyze
        
    Returns:
        Number of search queries found
    """
    pairs = re.findall(r"<begin_search>.*?</end_search>", text, re.DOTALL)
    return len(pairs)


def extract_boxed_answer(text: str) -> str:
    """
    Extract the answer from a LaTeX-style boxed format.
    
    Args:
        text: Text containing boxed content
        
    Returns:
        Extracted and cleaned content from the box or empty string if not found
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


def process_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single question item with search-augmented reasoning.
    
    Args:
        item: Question item containing 'id', 'question', and 'golden_answers'
        
    Returns:
        Dictionary containing the processed results
    """
    qid = item["id"]
    q_prompt = REASONING_PROMPT.format(question=item["question"])
    
    # Generate initial candidate answers
    candidate_answers, stop_reason = generate_chat_completion(
        q_prompt, n=3, stop_tokens=["</end_search>"]
    )
    
    for idx_, stop in enumerate(stop_reason):
        if stop is not None:
            candidate_answers[idx_] += "</end_search>\n\n"
            
    candidate_prompts = [q_prompt] * len(candidate_answers)
    candidate_flags = [True] * len(candidate_answers)
    turn_count = 0
    
    # Maximum number of search iterations
    MAX_TURNS = 10
    
    while any(candidate_flags) and turn_count < MAX_TURNS:
        for idx, (ans, current_prompt) in enumerate(zip(candidate_answers, candidate_prompts)):
            if not candidate_flags[idx]:
                continue
                
            thinking_part = ans.split("</think>")[0]
            search_query, ans_modified = check_search_token(thinking_part)
            
            if search_query == "":
                candidate_flags[idx] = False
                continue
                
            search_results = get_search_results(search_query)
            
            candidate_prompts[idx] += f"{ans_modified}\n\n{search_results}\n\n"
            
            new_ans_list, stop_reason = generate_completion(
                candidate_prompts[idx], n=1, stop_tokens=["</end_search>"]
            )
            
            if stop_reason[0] is not None:
                new_ans = new_ans_list[0] + "</end_search>\n\n"
            else:
                new_ans = new_ans_list[0]
                
            candidate_answers[idx] = new_ans
            
        turn_count += 1

    target = {
        "id": qid,
        "question": item["question"],
        "golden_answers": item["golden_answers"],
        "reasoning": [],
        "tool_usage": [],
        "boxed_answer": []
    }
    
    for ans, current_prompt in zip(candidate_answers, candidate_prompts):
        reasoning_prefix = current_prompt[len(q_prompt):]
        if "<begin_search>" in reasoning_prefix and "</end_search>" in reasoning_prefix:
            target["reasoning"].append(reasoning_prefix + ans)
            target["tool_usage"].append(count_search_nums(reasoning_prefix))
            target["boxed_answer"].append(extract_boxed_answer(ans))
        else:
            target["reasoning"].append(ans)
            target["tool_usage"].append(0)
            target["boxed_answer"].append(extract_boxed_answer(ans))
    
    return target


def process_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all questions in a document in parallel.
    
    Args:
        data: List of question items to process
        
    Returns:
        List of processed results
    """
    results = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(process_question, item): item for item in data}
        for future in tqdm(as_completed(futures), total=len(data), desc="Processing questions"):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"Exception processing question: {exc}")
    return results


def process_file(file_path: str, output_folder: str) -> str:
    """
    Process a single file: read data, generate answers, and write results.
    
    Args:
        file_path: Path to the input JSON file
        output_folder: Folder to save the output results
        
    Returns:
        Base name of the processed file
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    results = process_data(data)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Completed document: {file_path}")
    return base_name


def main():
    """
    Main function to process multiple document files in parallel.
    """
    start_time = time.time()
    
    # Configure paths
    dataset_folder_path = "./data/processed_data"
    output_folder = "./results/search_augmented_reasoning"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get all input files
    data_file_list = os.listdir(dataset_folder_path)
    file_paths = [os.path.join(dataset_folder_path, file_name) for file_name in data_file_list]
    
    # Process multiple documents in parallel
    with ThreadPoolExecutor(max_workers=3) as file_executor:
        futures = {
            file_executor.submit(process_file, file_path, output_folder): file_path 
            for file_path in file_paths
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
            try:
                base_name = future.result()
                print(f"Document {base_name} processing complete")
            except Exception as exc:
                print(f"Exception processing document: {exc}")
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()