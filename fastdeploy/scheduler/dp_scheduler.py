from fastdeploy.scheduler.local_scheduler import LocalScheduler
from multiprocessing import Queue
import threading
from typing import Dict, List, Optional, Tuple
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.scheduler.data import ScheduledRequest, ScheduledResponse

class DPScheduler:
    def __init__(self,
        max_size: int,
        ttl: int,
        enable_chunked_prefill: bool,
        max_num_partial_prefills: int,
        max_long_partial_prefills: int,
        long_prefill_token_threshold: int
        ):
        self._scheduler = LocalScheduler(max_size, ttl, enable_chunked_prefill, max_num_partial_prefills, max_long_partial_prefills, long_prefill_token_threshold)
        
    def start(self, dp_rank: int, request_queues: List[Queue], result_queue: Queue):
        self.dp_rank = dp_rank
        self.request_queues = request_queues
        self.result_queue = result_queue
        threading.Thread(target=self._put_requests_to_local).start()
        threading.Thread(target=self._get_response_from_local).start()

    def put_requests(self, requests: List[Dict]):
        results = []
        for request in requests:
            self.request_queues[request.dp_rank].put(request)
            results.append((request.request_id, None))
        return results
    
    def _put_requests_to_local(self):
        while True:
            request = self.request_queues[self.dp_rank].get()
            self._scheduler.put_requests([request])
    
    def _get_response_from_local(self):
        while True:
            results = self._scheduler.get_results()
            if len(results) == 0:
                continue
            self.result_queue.put(results)

    def get_requests(
        self,
        available_blocks,
        block_size,
        reserved_output_blocks,
        max_num_batched_tokens,
        batch=1,
    ) -> List[Request]:
        return self._scheduler.get_requests(available_blocks, block_size, reserved_output_blocks, max_num_batched_tokens, batch)
    
    def put_results(self, results: List[RequestOutput]):
        self._scheduler.put_results(results)
    
    
    def get_results(self) -> Dict[str, List[RequestOutput]]:
        return self.result_queue.get()
    
