"""Label-free pooling of complementary binary-event forecasts."""
import math
from .domain import checked_probability

def logit(p):
    p=min(1-1e-6,max(1e-6,checked_probability(p)))
    return math.log(p/(1-p))

def fuse(a,b,method):
    """a forecasts correct; b forecasts incorrect. No labels are accepted."""
    a=checked_probability(a);b=checked_probability(b)
    if method=='arithmetic':return (a+1-b)/2
    if method=='logit':return 1/(1+math.exp(-(logit(a)-logit(b))/2))
    if method=='normalized_support':return a/(a+b) if a+b else .5
    raise ValueError('Unknown complement fusion method')
