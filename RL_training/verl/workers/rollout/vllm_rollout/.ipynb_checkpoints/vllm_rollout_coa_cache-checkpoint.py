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
import os
import hashlib
from collections import OrderedDict
import threading
# atexit

import atexit

BASE_URL = "http://47.94.155.236:8080/retrieve"

class RetrievalCache:
    def __init__(self, max_size=100000, ttl=36000, cache_dir="/root/autodl-tmp/project/verl/api_cache", 
                 cache_file="retrieval_cache.json", auto_save_interval=30):
        """
        初始化检索缓存
        
        Args:
            max_size: 缓存最大条目数
            ttl: 缓存条目的生存时间(秒)
            cache_dir: 缓存文件存储目录
            cache_file: 缓存文件名
            auto_save_interval: 自动保存间隔(秒)，0表示禁用自动保存
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
        # 持久化相关设置
        self.cache_dir = cache_dir
        self.cache_file = cache_file
        self.cache_path = os.path.join(cache_dir, cache_file)
        self.auto_save_interval = auto_save_interval
        self._last_save_time = time.time()
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 从本地加载缓存
        self.load_cache()
        
        # 注册程序退出时保存缓存的钩子
        atexit.register(self.save_cache)
        
        # 如果启用了自动保存，开启自动保存线程
        if auto_save_interval > 0:
            self._start_auto_save()
    
    def _get_key(self, text, top_k, return_score):
        """生成缓存键"""
        key_str = f"{text}_{top_k}_{return_score}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, text, top_k, return_score):
        """从缓存获取结果"""
        key = self._get_key(text, top_k, return_score)
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp <= self.ttl:
                    # 将访问的项移到末尾（最新使用）
                    self.cache.move_to_end(key)
                    self.hits += 1
                    return value
                else:
                    # 已过期，删除
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def put(self, text, top_k, return_score, result):
        """添加结果到缓存"""
        key = self._get_key(text, top_k, return_score)
        
        with self.lock:
            # 如果缓存已满，删除最早的项
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (result, time.time())
            
            # 检查是否应该自动保存
            if self.auto_save_interval > 0 and time.time() - self._last_save_time > self.auto_save_interval:
                self._save_cache_async()
    
    def get_stats(self):
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_ratio = self.hits / total if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio,
            "cache_file": self.cache_path
        }
    
    def load_cache(self):
        """从本地文件加载缓存"""
        try:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                current_time = time.time()
                with self.lock:
                    # 清空当前缓存
                    self.cache.clear()
                    
                    # 只加载未过期的条目
                    loaded_count = 0
                    for key, (value, timestamp) in cache_data.items():
                        if current_time - timestamp <= self.ttl and loaded_count < self.max_size:
                            self.cache[key] = (value, timestamp)
                            loaded_count += 1
                
                print(f"已从 {self.cache_path} 加载 {loaded_count} 条缓存数据")
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
    
    def save_cache(self):
        """将缓存保存到本地文件"""
        try:
            with self.lock:
                # 将OrderedDict转换为普通dict以便序列化
                cache_data = {k: v for k, v in self.cache.items()}
                
                # 确保目录存在
                import fcntl
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    json.dump(cache_data, f, ensure_ascii=False)
                with open(self.cache_path, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False)
                
                self._last_save_time = time.time()
                print(f"已将 {len(self.cache)} 条缓存数据保存到 {self.cache_path}")
        except Exception as e:
            print(f"保存缓存文件失败: {e}")
    
    def _save_cache_async(self):
        """在后台线程中保存缓存"""
        threading.Thread(target=self.save_cache, daemon=True).start()
    
    def _start_auto_save(self):
        """启动自动保存线程"""
        def auto_save():
            while True:
                time.sleep(self.auto_save_interval)
                print(self.get_stats())
                self._save_cache_async()
        
        threading.Thread(target=auto_save, daemon=True).start()
        print(f"已启动自动保存缓存线程，间隔 {self.auto_save_interval} 秒")
    
    def clear_cache(self):
        """清空缓存并删除缓存文件"""
        with self.lock:
            self.cache.clear()
            if os.path.exists(self.cache_path):
                try:
                    os.remove(self.cache_path)
                    print(f"已删除缓存文件 {self.cache_path}")
                except Exception as e:
                    print(f"删除缓存文件失败: {e}")

# 创建缓存实例
retrieval_cache = RetrievalCache(max_size=100000, ttl=18000000) 

# 带重试机制的检索函数
def do_retrevial(text, top_k=3, return_score=True, max_retries=2, retry_delay=0.5):
    """
    执行检索，带缓存和重试机制
    
    Args:
        text: 查询文本
        top_k: 返回结果数量
        return_score: 是否返回分数
        max_retries: 最大重试次数
        retry_delay: 重试间隔(秒)
    
    Returns:
        检索结果字典或None
    """
    # 先查询缓存
    cached_result = retrieval_cache.get(text, top_k, return_score)
    if cached_result is not None:
        return cached_result
    
    payload = {"query": text, "tok_k": top_k, "return_score": return_score}
    
    retries = 0
    while retries <= max_retries:
        try:
            start_time = time.time()
            response = requests.post(BASE_URL, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            # 验证返回结果格式
            assert isinstance(result, dict), "Response should be a dict"
            if return_score:
                assert len(result) == 2, "Each item should be {\"documents\": [...], \"scores\": [...]}"
            else:
                assert len(result) == 1, "Each item should be {\"documents\": [...]}"
            
            # 缓存结果
            retrieval_cache.put(text, top_k, return_score, result)
            return result
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, AssertionError) as e:
            retries += 1
            if retries <= max_retries:
                print(f"Retrieval attempt {retries} failed: {e}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避
            else:
                print(f"Retrieval failed after {max_retries} attempts: {e}")
                return None

def get_cache_stats():
    """获取缓存统计信息"""
    return retrieval_cache.get_stats()

# 原有的topk_format函数保持不变
def topk_format(result, topk=1):
    if result is None:
        return "Retrieval service unavailable."
    
    if topk == 1:
        return result["documents"][0]["contents"] 
    else:
        content_list = []
        for index, doc in enumerate(result["documents"], start=1):
            content_list.append(f"result {index}: {doc['contents']}")

        final_result = "\n".join(content_list)
        return final_result

def get_search_results(text):
    search_reslut_token = "<search_result> {search_result} </search_result>"
    
    retrevial_result = do_retrevial(text, top_k=3)
    if retrevial_result is None:
        return search_reslut_token.format(search_result="Retrieval service unavailable.")
    
    search_result = topk_format(retrevial_result, topk=3)
    
    return search_reslut_token.format(search_result=search_result)

# 定期清理过期缓存条目的函数(可选)
def clean_expired_cache_entries():
    current_time = time.time()
    with retrieval_cache.lock:
        keys_to_remove = []
        for key, (_, timestamp) in retrieval_cache.cache.items():
            if current_time - timestamp > retrieval_cache.ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del retrieval_cache.cache[key]

# Format the generated text
def check_search_token(text):
    pattern = r"<begin_search>(.*?)</end_search>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[:match.end()]
    return "", ""

# def count_search_nums(text):
#     pairs = re.findall(r"<begin_search>.*?</end_search>", text, re.DOTALL)
#     return len(pairs)



def process_search_answer(ans, current_prefix,reach_limit=False):
    
    search_query, ans_modified = check_search_token(ans.split("</think>")[0])
    if search_query == "":
        return current_prefix, False
    
    # print(f"需要检索的问题是 {search_query}")
    # print("-------searching...-------")
    if reach_limit:
        new_prefix = current_prefix + f"{ans_modified}\n\n<search_result> Reach the limit of search times. </search_result>\n\n"
        return new_prefix, True
        
    search_results = get_search_results(search_query)
    # print(f"问题检索的答案是 {search_results}\n\n")
    new_prefix = current_prefix + f"{ans_modified}\n\n{search_results}\n\n"
    return new_prefix, True

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


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

        # users can customize different sampling_params at different run
        # </end_search> is the stop string for search token
        currnet_n = 1
        # print(kwargs)
        # print(self.sampling_params)
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
        # kwargs.pop('stop', None)
        # print(kwargs)
        # print(self.sampling_params)
        print(f"sampling time: {time.time() - start_time}")

        response = output[0].tolist()
        
        
        # max_search_nums = self.config.tool_call_limit
        max_search_nums = 6
        
        prompt_len = [len(i) for i in idx_list]
        
        new_prefix_ids = []
        

        
        current_prefix_list = []
        for sample_idx in range(len(idx_list)):
            for _n in range(current_n):
                current_prefix_list.append(idx_list[sample_idx])
        
        current_prefix_list = self.tokenizer.batch_decode(current_prefix_list,skip_special_tokens=False)
        response_str_list = self.tokenizer.batch_decode(response,skip_special_tokens=False)
        raw_current_prefix_list = copy.deepcopy(current_prefix_list)
        # print(f"current_prefix_list: {current_prefix_list[0]}")
        # print(f"response_str_list: {response_str_list[0]}")
        
       
        # 复制一份sampling_params，用于搜索，这一份不能影响原来的sampling_params
        re_sampling_params = copy.deepcopy(self.sampling_params)
        re_sampling_params.n = 1
        re_sampling_params.max_tokens=1024
        re_sampling_params.stop_token_ids=[eos_token_id]
        # for iter in tqdm(range(max_search_nums + 1),desc="Searching...",disable=False):
        print('re_sampling_params', re_sampling_params)
        pber = tqdm(range(max_search_nums + 1),desc="Searching...",disable=False)        
        for iter in pber:
                
            assert len(response_str_list) == len(current_prefix_list), "response_str_list and current_prefix_list should have the same length,but got {} and {}".format(len(response_str_list), len(current_prefix_list))
            
            start_time = time.time()
            
            new_prefix_list = []
            search_flag_list = []
            if iter == max_search_nums:
                re_sampling_params.max_tokens=2048
            for response_str, current_prefix in zip(response_str_list, current_prefix_list): 
                new_prefix, search_flag = process_search_answer(response_str, current_prefix, reach_limit=iter == max_search_nums)
                
                new_prefix_list.append(new_prefix)
                search_flag_list.append(search_flag)
            
            current_prefix_list = new_prefix_list
            # pber.set_description(f"Done for Searching...{iter+1}, now re-generating {sum(search_flag_list)}/{len(search_flag_list)} samples")
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
                # print(f"|**|input_ids_list: {input_ids_list[0]} for search{iter}")
                # print(f"|**|new_prompts_list: {new_prompts_list[0]}")
                
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
        # print(f"full_content: {full_content[0]}")

        # full_content_ids = []
        response = []
        
        for p_id, p in enumerate(full_content):
            response.append(self.tokenizer.encode(p[len(raw_current_prefix_list[p_id]):], add_special_tokens=False))
            with open("/root/autodl-tmp/project/verl/full_content.txt", "a",encoding="utf-8") as f:
                f.write("full_content_last_round: {}\n\n\n".format(p[len(raw_current_prefix_list[p_id]):].split("｜end▁of▁sentence｜")[0]))    

       
        # response = []
        # for i in range(len(full_content_ids)):
        #     response.append(full_content_ids[i][len(idx_list[i//self.config.n]):])
        
        

        # 将嵌套列表转为tensor
        
        # 删除 response尾部的eos，最后只保留一个eos
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

                # 截断？ 多轮对话后长度可能大于原来设定的max_tokens
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
