import unittest,math
from llm_kt.complement import fuse
from llm_kt.providers import Chat
class ComplementTests(unittest.TestCase):
 def test_consistency(self):
  for p in [.001,.2,.5,.9,.999]:
   for m in ['arithmetic','logit','normalized_support']:self.assertAlmostEqual(fuse(p,1-p,m),p)
 def test_swap(self):
  for m in ['arithmetic','logit','normalized_support']:self.assertAlmostEqual(fuse(.8,.3,m),1-fuse(.3,.8,m))
 def test_bias_cancel(self):
  sig=lambda x:1/(1+math.exp(-x))
  self.assertAlmostEqual(fuse(sig(1.2+.7),sig(-1.2+.7),'logit'),sig(1.2))
 def test_bounds(self):
  for m in ['arithmetic','logit','normalized_support']:
   self.assertTrue(0<=fuse(0,0,m)<=1);self.assertTrue(0<=fuse(1,1,m)<=1)
  with self.assertRaises(ValueError):fuse(-1,.2,'arithmetic')
 def test_chat_inverse(self):
  cfg={'probability_strategy':'decimal_probability','prompt_snapshot':{'output_event':'incorrect'}}
  r={'choices':[{'finish_reason':'stop','message':{'content':'0.2000'}}]}
  self.assertAlmostEqual(Chat(cfg).probability(r),.8)
  cfg['prompt_snapshot']['output_event']='correct';self.assertAlmostEqual(Chat(cfg).probability(r),.2)
if __name__=='__main__':unittest.main()
