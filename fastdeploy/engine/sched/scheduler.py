import time
from collections import defaultdict, deque
from collections.abc import Iterable
from typing import Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import threading

from fastdeploy.engine.request import Request, RequestStatus
from fastdeploy.cache_manager.prefix_cache_manager import PrefixCacheManager

class Scheduler:
    def __init__(self,
                 max_num_seqs,
                 config,
                 tensor_parallel_size,
                 splitwise_role,
                 local_data_parallel_id=0
        ):
        self.config = config
        self.max_num_seqs = max_num_seqs
        self.stop_flags = [True] * max_num_seqs
        self.real_bsz = 0
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.cache_manager = PrefixCacheManager(config, tensor_parallel_size,
                                                splitwise_role,
                                                local_data_parallel_id)
        self.finish_execution_pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
    

    def available_batch(self):
        """
        available batch size for engine

        Returns:
            int: available batch size
        """
        return np.sum(self.stop_flags)

    def available_block_num(self):
        """
        available block size for engine

        Returns:
            int: available block size
        """
        return len(self.cache_manager.gpu_free_block_list)
    
    def allocated_slots(self, request: Request):
        return len(request.block_tables) * self.config.block_size
    
    def get_new_block_nums(self, request: Request, num_new_tokens: int):
        return (request.num_computed_tokens + num_new_tokens + self.config.block_size - 1) // self.config.block_size - len(request.block_tables)

    def schedule(self):
        with self.lock:
            scheduled_new_reqs: list[Request] = []
            scheduled_resumed_reqs: list[Request] = []
            scheduled_running_reqs: list[Request] = []
            preempted_reqs: list[Request] = []
            token_budget = self.config.max_num_scheduled_tokens

            # First, schedule the RUNNING requests.
            req_index = 0
            num_decoding_req_nums = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.num_computed_tokens >= request.prompt_token_ids_len: # 在Decoding
                    request.num_computed_tokens = request.num_total_tokens - 1
                    if self.allocated_slots(request) - request.num_total_tokens <= self.config.cache_config.prealloc_dec_block_slot_num_threshold:
                        # 需要分配下一次的解码block
                        if self.cache_manager.can_allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num):
                            # 分配解码下一轮的解码 block
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num))
                            # 进入running list
                            scheduled_running_reqs.append(request)
                        else:
                            # 触发抢占
                            while True:
                                if self.cache_manager.can_allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num):
                                    preempted_req = self.running.pop()
                                    self.kv_cache_manager.free(preempted_req)
                                    preempted_req.status = RequestStatus.PREEMPTED
                                    preempted_req.num_computed_tokens = 0
                                    self._free_blocks(preempted_req)  # 由于异步存在，抢占请求需要让推理不再推
                                    self.waiting.appendleft(preempted_req)
                                    preempted_reqs.append(preempted_req)
                                    if preempted_req == request:
                                        # No more request to preempt.
                                        can_schedule = False
                                        break
                                else:
                                    # The request can be scheduled.
                                    can_schedule = True
                                    break
                            if not can_schedule:
                                break
                            # 分配解码下一轮的解码 block
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num))
                            # 进入running list
                            scheduled_running_reqs.append(request)   
                        num_decoding_req_nums += 1
                        token_budget -= 1
                else:  # 在Prefill
                    num_new_tokens = request.num_total_tokens - request.num_computed_tokens
                    num_new_tokens = min(num_new_tokens, token_budget)
                    new_new_block = self.get_new_block_nums(request, num_new_tokens)
                    # 需要分配下一次的prefill block
                    if self.cache_manager.can_allocate_gpu_blocks(new_new_block):
                        # 分配解码下一轮的解码 block
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(new_new_block))
                        # 进入running list
                        scheduled_running_reqs.append(request)
                    else:
                        # 触发抢占
                        while True:
                            if self.cache_manager.can_allocate_gpu_blocks(new_new_block):
                                preempted_req = self.running.pop()
                                self.kv_cache_manager.free(preempted_req)
                                preempted_req.status = RequestStatus.PREEMPTED
                                preempted_req.num_computed_tokens = 0
                                self._free_blocks(preempted_req)  # 由于异步存在，抢占请求需要让推理不再推
                                self.waiting.appendleft(preempted_req)
                                preempted_reqs.append(preempted_req)
                                if preempted_req == request:
                                    # No more request to preempt.
                                    can_schedule = False
                                    break
                            else:
                                # The request can be scheduled.
                                can_schedule = True
                                break
                        if not can_schedule:
                            break
                        # 分配解码下一轮的解码 block
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(new_new_block))
                        # 进入running list
                        scheduled_running_reqs.append(request) 
                    token_budget -= num_new_tokens
                    request.num_computed_tokens += num_new_tokens
                req_index += 1
            # Next, schedule the WAITING requests.
            if not preempted_reqs:
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_seqs:
                        break
                    request = self.waiting[0]
                    if request.status == RequestStatus.WAITING:
                        num_new_tokens = request.num_total_tokens - request.num_computed_tokens
                        num_new_tokens = min(num_new_tokens, token_budget)
                        new_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # 需要分配下一次的prefill block
                        if self.cache_manager.can_allocate_gpu_blocks(new_new_block):
                            # 分配解码下一轮的解码 block
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(new_new_block))
                            self.waiting.popleft()
                            self.running.append(request)
                            # scheduled_new_reqs list
                            scheduled_new_reqs.append(request)
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            request.status = RequestStatus.RUNNING
                            allocated_position = self.get_available_position()
                            request.idx = allocated_position
                            self.tasks_list[allocated_position] = request
                            self.stop_flags[allocated_position] = False
                        else:
                            break
                    elif request.status == RequestStatus.PREEMPTED:
                        num_new_tokens = request.num_total_tokens - request.num_computed_tokens
                        num_new_tokens = min(num_new_tokens, token_budget)
                        new_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # 需要分配下一次的prefill block
                        if self.cache_manager.can_allocate_gpu_blocks(new_new_block):
                            # 分配解码下一轮的解码 block
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(new_new_block))
                            self.waiting.popleft()
                            self.running.append(request)
                            # scheduled_resumed_reqs list
                            scheduled_resumed_reqs.append(request)
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            request.status = RequestStatus.RUNNING
                        else:
                            break
            # Todo: 处理请求字段，看如何和引擎互动

        
    def get_available_position(self) -> int:
        position = 0
        while position < self.max_num_seqs:
            if self.stop_flags[position] is True:
                return position
        assert True is False, "No available position is available for new request"

    def get_real_bsz(self) -> int:
        for i in range(self.max_num_seqs - 1, -1, -1):
            if not self.stop_flags[i]:
                self.real_bsz = i + 1
                break
        return self.real_bsz
    
    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request
    
    def _free_blocks(request: Request):
        self.cache_manager.recycle_gpu_blocks(request.block_tables)
        request.block_tables = []
    
    def finish_requests_async(self, 
        request_ids: Union[str, Iterable[str]]):
        self.finish_execution_pool.submit(self.finish_requests, request_ids)
    
    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]]):
        with self.lock:
            if isinstance(request_ids, str):
                request_ids = (request_ids, )
            else:
                request_ids = set(request_ids)

            for req_id in request_ids:
                request = self.requests.get(req_id)
                if request is None:
                    # Invalid request ID.
                    continue
                request.status = RequestStatus.FINISHED
                self.running.remove(request)
                self._free_blocks(request)
                self.tasks_list[request.idx] = None
                self.stop_flags[request.idx] = False
                del self.requests[req_id]
            

             
        
    

