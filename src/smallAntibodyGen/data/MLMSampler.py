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
        """
        Update the epoch-dependent shuffle seed used by the sampler.

        Args:
            epoch:
                Zero-based epoch index.

        Returns:
            None.
        """
        self.epoch = epoch

    def _record_bucket_key(self, record) -> tuple[str, int]:
        """
        Compute the batching bucket for one dataset record.

        For classic single-chain records, `chain_group` will be `"heavy"` or
        `"light"`. For native paired examples it will be `"paired"`. We bucket
        on token length when available because paired examples carry separator
        and chain-marker tokens that materially affect padding cost.

        Args:
            record:
                Dataset record exposing `chain_group`, `token_length`, and
                `length`.

        Returns:
            Tuple `(group_name, length_bucket)` used as the dictionary key.
        """
        effective_length = getattr(record, "token_length", None) or getattr(record, "length")
        length_bucket = int(effective_length) // self.bucket_width
        return (record.chain_group, length_bucket)
        
    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices grouped by chain type and rough sequence length.

        Returns:
            Iterator of lists of dataset indices, one batch at a time.
        """
        rng = random.Random(self.seed + self.epoch)
        
        buckets = defaultdict(list)
        
        for idx, record in enumerate(self.dataset.records):
            bucket_key = self._record_bucket_key(record)
            buckets[bucket_key].append(idx)
        
        all_batches: List[List[int]] = []
        
        for bucket_key, indices in buckets.items():
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start: start + self.batch_size] # randomly choosing batches
                if len(batch) == self.batch_size or not self.drop_last: # if the batch is now the same size or if you don't drop the last one
                    all_batches.append(batch)
        
        rng.shuffle(all_batches)
        yield from all_batches # select one batch at a time from all batches
        
    def __len__(self) -> int:
        """
        Return the number of batches the sampler will emit for one epoch.

        Returns:
            Integer number of batches.
        """
        total = 0
        buckets = defaultdict(int)
        
        for record in self.dataset.records:
            bucket_key = self._record_bucket_key(record)
            buckets[bucket_key] += 1
            
        for count in buckets.values():
            if self.drop_last:
                total += count // self.batch_size
            else: 
                total += math.ceil(count / self.batch_size)
                
        return total
