# Copyright 2025 the LlamaFactory team.
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

import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template
from ...extras import logging
logger = logging.get_logger(__name__)

@dataclass
class DatasetProcessor(ABC):
    r"""
    A class for data processors.
    """

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        r"""
        Builds model inputs from the examples.
        """
        ...

    @abstractmethod
    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        r"""
        Print a data example to stdout.
        """
        ...


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


# def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
#     r"""
#     Computes the real sequence length after truncation by the cutoff_len.
#     """
#     if target_len * 2 < cutoff_len:  # truncate source
#         max_target_len = cutoff_len
#     elif source_len * 2 < cutoff_len:  # truncate target
#         max_target_len = cutoff_len - source_len
#     else:  # truncate both
#         max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

#     new_target_len = min(max_target_len, target_len)
#     max_source_len = max(cutoff_len - new_target_len, 0)
#     new_source_len = min(max_source_len, source_len)
#     return new_source_len, new_target_len

def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    truncated = False
    truncation_type = None
    
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
        truncation_type = "source"
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
        truncation_type = "target"
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
        truncation_type = "both"

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    
    truncated = (new_source_len < source_len) or (new_target_len < target_len)
    if truncated:
        logger.info(
            f"Truncation occurred ({truncation_type}): source {source_len}->{new_source_len}, "
            f"target {target_len}->{new_target_len}, total {source_len + target_len}->{new_source_len + new_target_len}, "
            f"cutoff_len {cutoff_len}"
        )
    
    return new_source_len, new_target_len
