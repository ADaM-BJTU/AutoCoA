# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

import copy


import re

import requests
import json
import time
import asyncio
import httpx

BASE_URL = "http://localhost:8080/retrieve"

# Call tool function
def do_retrevial(text, top_k=3, return_score=True):
    payload = {"query": text, "tok_k": top_k, "return_score": return_score}
    
    try:
        start_time = time.time()
        response = requests.post(BASE_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        # print(result)
        # print(f"Single Query (return_score={return_score}):")
        # print(json.dumps(result, indent=2))
        # print(f"Time taken: {time.time() - start_time:.2f} seconds")
        # print(type(result))
        assert isinstance(result, dict), "Response should be a dict"
        if return_score:
            assert len(result) == 2, "Each item should be {\"documents\": [...], \"scores\": [...]}"
        else:
            assert len(result) == 1, "Each item should be {\"documents\": [...]}"
        # print("Single query test passed!")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Single query failed: {e}")
        return None
def do_batch_retrevial(text_list, top_k=3, return_score=True, batch_size=512):
    result = []

    payload = {"query": text_list, "tok_k": top_k, "return_score": return_score}
    retries = 2
    while retries > 0:
        try:
            response = requests.post(BASE_URL, json=payload, timeout=20)
            response.raise_for_status()
            result.extend(response.json())
            
            break
        except requests.exceptions.RequestException as e:
            print(f"Batch query failed: {e}")
            retries -= 1
            if retries == 0:
                result.extend([None] * len(text_list))
            continue
        
    return result
    

def topk_format(result, topk=1):
    if topk == 1:
        return result["documents"][0]["contents"] 
    else:
        content_list = []
        for index, doc in enumerate(result["documents"], start=1):
            content_list.append(f"result {index}: {doc['contents']}")

        final_result = "\n".join(content_list)
        return final_result
def get_search_results(texts):
    
    # search_reslut_token = "<search_result> {search_result} </search_result>"
    

    retrevial_result = do_batch_retrevial(texts, top_k=3)
    
    search_result_list = []
    for res in retrevial_result:
        if res is None:
            # search_result_list.append("Retrevial failed.")
            search_reslut_token = "<search_result> Retrevial failed. </search_result>\n\n"
            search_result_list.append(search_reslut_token)
        else:
            # search_result_list.append(topk_format(res, topk=3))
            search_reslut_token = "<search_result> " + topk_format(res, topk=3) + " </search_result>\n\n"
            search_result_list.append(search_reslut_token)
            
    
        
    return search_result_list
    



    
# Format the generated text
def check_search_token(text):
    pattern = r"<begin_search>(.*?)</end_search>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[:match.end()]
    return "", ""



def process_search_answer_batch(ans_list, current_prefix_list, reach_limit=False):
    search_queries = []
    ans_modified_list = []
    for ans in ans_list:
        search_query, ans_modified = check_search_token(ans.split("</think>")[0])
        search_queries.append(search_query)
        ans_modified_list.append(ans_modified)
    
    new_prefix_list = [None] * len(current_prefix_list)  
    search_flag_list = [False] * len(current_prefix_list)  
    
    for i, search_query in enumerate(search_queries):
        if search_query == "":
            new_prefix_list[i] = current_prefix_list[i]  
            search_flag_list[i] = False
        else:
            if reach_limit:
                new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Reach the limit of search times. </search_result>\n\n"
                new_prefix_list[i] = new_prefix  
                search_flag_list[i] = True
            else:
                search_flag_list[i] = True
    
    if not reach_limit and any(search_flag_list):
        try:
            queries_to_search = []
            query_indices = []  
            for i, flag in enumerate(search_flag_list):
                if flag:
                    queries_to_search.append(search_queries[i])
                    query_indices.append(i)
            
            search_results = get_search_results(queries_to_search)
            
            assert len(search_results) == len(queries_to_search), f"搜索结果数量与查询数量不匹配: {len(search_results)} vs {len(queries_to_search)}"
            
            for idx, result_idx in enumerate(query_indices):
                if idx < len(search_results):  
                    new_prefix = current_prefix_list[result_idx] + f"{ans_modified_list[result_idx]}\n\n{search_results[idx]}\n\n"
                    new_prefix_list[result_idx] = new_prefix  
        
        except Exception as e:
            print("An error occurred during the search process: ", e)
            for i in range(len(search_queries)):
                if search_flag_list[i] and new_prefix_list[i] is None:
                    new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Retrieval failed due to an error. </search_result>\n\n"
                    new_prefix_list[i] = new_prefix
    
    return new_prefix_list, search_flag_list
    

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer
        
        self.max_search_nums = config.get('max_search_nums', 10)
        
    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        import time
        start_time = time.time()
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        currnet_n = 1
        with self.update_sampling_params(**kwargs):
            current_n = self.sampling_params.n
            tmp_sampling_params = copy.deepcopy(self.sampling_params)
            tmp_sampling_params.max_tokens = 2048
            tmp_sampling_params.stop_token_ids = [eos_token_id]
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=tmp_sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)
        print(f"sampling time: {time.time() - start_time}")

        response = output[0].tolist()
        
        
        
        
        
        current_prefix_list = []
        for sample_idx in range(len(idx_list)):
            for _n in range(current_n):
                current_prefix_list.append(idx_list[sample_idx])
        
        current_prefix_list = self.tokenizer.batch_decode(current_prefix_list,skip_special_tokens=False)
        response_str_list = self.tokenizer.batch_decode(response,skip_special_tokens=False)
        raw_current_prefix_list = copy.deepcopy(current_prefix_list)
        
       
        re_sampling_params = copy.deepcopy(self.sampling_params)
        re_sampling_params.n = 1
        re_sampling_params.max_tokens=1024
        re_sampling_params.stop_token_ids=[eos_token_id]

        print('re_sampling_params', re_sampling_params)
        pber = tqdm(range(self.max_search_nums + 1),desc="Searching...",disable=False)        
        for iter in pber:
                
            assert len(response_str_list) == len(current_prefix_list), "response_str_list and current_prefix_list should have the same length,but got {} and {}".format(len(response_str_list), len(current_prefix_list))
            
            
            new_prefix_list = []
            search_flag_list = []
            if iter == self.max_search_nums:
                re_sampling_params.max_tokens=2048
            start_time = time.time()
            new_prefix_list, search_flag_list = process_search_answer_batch(response_str_list, current_prefix_list, reach_limit=iter == self.max_search_nums)
            
            current_prefix_list = new_prefix_list
            pber.set_description(f"Done for Searching...{iter+1}, cost time: {time.time() - start_time:.2f}s, now re-generating {sum(search_flag_list)}/{len(search_flag_list)} samples")
            
            if any(search_flag_list):
                new_prompts_list = []
                for search_flag, current_prefix in zip(search_flag_list, current_prefix_list):
                    if search_flag:
                        new_prompts_list.append(current_prefix)
                    else:
                        pass
                
                input_ids_list = []
                
                for p in new_prompts_list:
                    input_ids_list.append(self.tokenizer.encode(p, add_special_tokens=False))
                
                new_response_list = self.inference_engine.generate(
                    prompts=None,  # because we have already convert it to prompt token id
                    sampling_params=re_sampling_params,
                    prompt_token_ids=input_ids_list,
                    use_tqdm=False)[0]
                  
                new_response_str_list = self.tokenizer.batch_decode(new_response_list,skip_special_tokens=False)
                
                for i, search_flag in enumerate(search_flag_list):
                    if search_flag:
                        response_str_list[i] = new_response_str_list.pop(0)        

            else:
                break
            
        pber.close()
            

        full_content = []
        for response_str, current_prefix in zip(response_str_list, current_prefix_list):
            full_content.append(current_prefix + response_str)
        response = []
        
        for p_id, p in enumerate(full_content):
            response.append(self.tokenizer.encode(p[len(raw_current_prefix_list[p_id]):], add_special_tokens=False))

        max_response_len = -1


        for i_res in range(len(response)):
            while len(response[i_res])> 1 and response[i_res][-1] == eos_token_id:
                if response[i_res][-2] == eos_token_id:
                    response[i_res] = response[i_res][:-1]
                else:
                    break
                
                
            max_response_len = max(max_response_len, len(response[i_res]))
            
            # response[i_res] = torch.tensor(response[i_res], device=attention_mask.device, dtype=attention_mask.dtype)
        
        # pad right to largest response
                
        with self.update_sampling_params(**kwargs):
                    # max_tokens
            # print(f"max_response_len: {max_response_len}")
            # print(f"sampling_params.max_tokens: {self.sampling_params.max_tokens}")
            for i_res in range(len(response)):
                response[i_res] = response[i_res] + [eos_token_id]*(max_response_len - len(response[i_res]))

                response[i_res] = response[i_res][:self.sampling_params.max_tokens]
                
        
        response = torch.tensor(response, device=attention_mask.device, dtype=attention_mask.dtype)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        # response = output[0].to(idx.device)
        # log_probs = output[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        
        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)
