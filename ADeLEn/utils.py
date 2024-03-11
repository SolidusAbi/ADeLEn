from collections import deque
from itertools import islice

def slide(iterable, size):
    '''
        Iterate through iterable using a sliding window of several elements.
        Important: It is a generator!.
        
        Creates an iterable where each element is a tuple of `size`
        consecutive elements from `iterable`, advancing by 1 element each
        time. For example:
        >>> list(sliding_window_iter([1, 2, 3, 4], 2))
        [(1, 2), (2, 3), (3, 4)]
        
        source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
    '''
    iterable = iter(iterable)
    window = deque(islice(iterable, size), maxlen=size)
    for item in iterable:
        yield tuple(window)
        window.append(item)
    if window:  
        # needed because if iterable was already empty before the `for`,
        # then the window would be yielded twice.
        yield tuple(window)


from torch import nn
class SkipConnectionSequential(nn.Sequential):
    '''
        Sequential module return the output of the last layer and the intermediate
        for the skip connections.
    '''
    def __init__(self, *args) -> None:
        super(SkipConnectionSequential, self).__init__(*args)

    def forward(self, x) -> tuple:
        ''' 
            Returns
            -------
            x: torch.Tensor,
                The output of the encode path.

            sk: list,
                A list of the intermediate outputs of the encode path, representing
                the skip connections. This list is ordered by the depth of the
                skip connections, where the first element is the deepest and the
                last element is the shallowest.
        '''

        sk = []
        for module in self:
            module, down = module[:-1], module[-1]
            x = module(x)
            sk.append(x.clone().detach())
            x = down(x)        
        
        sk.reverse()
        return x, sk