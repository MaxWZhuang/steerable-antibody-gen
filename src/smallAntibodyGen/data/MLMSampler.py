import math
import random
from collections import defaultdict
from typing import Iterator, List

from torch.utils.data import Sampler

class ChainLengthBucketBatchSampler(Sampler[List[int]]):
    """
    Creates batches that are 
    1) Chain-homogenous
    2) Roughly length-bucketed
    
    This prevents sequences of very different lengths from being padded together (in general cases) and 
    that heavy and light chains are not mixed. 
    """
    
    def __init__(
        self, 
        dataset,
        batch_size: int, 
        bucket_width: int = 8, 
        drop_last: bool = False,
        seed: int = 42
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        
    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        
        buckets = defaultdict(list)
        
        for idx, record in enumerate(self.dataset.records):
            length_bucket = record.length // self.bucket_width # floor div
            bucket_key = (record.chain_group, length_bucket)
            buckets[bucket_key].append(idx)
        
        all_batches: List[List[int]] = []
        
        for bucket_key, indices in buckets.items():
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start: start + self.batch_size] # randomly choosing batches
                if len(batch) == self.batch_size or not self.drop_last: # if the batch is now the same size or if you don't drop the last one
                    all_batches.append[batch]
        
        rng.shuffle(all_batches)
        yield from all_batches # select one batch at a time from all batches
        
    def __len__(self) -> int:
        total = 0
        buckets = defaultdict(int)
        
        for record in self.dataset.records:
            length_bucket = record.length // self.bucket_width
            bucket_key = (record.chain_group, length_bucket)
            buckets[bucket_key] += 1
            
        for count in buckets.values():
            if self.drop_last:
                total += count // self.batch_size
            else: 
                total += math.ceil(count / self.batch_size)
                
        return total