from torch.utils.data import Sampler
import random
class randomsampler(Sampler):
    def __len__(self):
        return 2147483647;
    def __iter__(self) :
        for i_batch in range(2147483647):
            yield random.randint(0,2147483647);
