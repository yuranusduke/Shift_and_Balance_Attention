"""
This file builds Shift-and-Balance Attention(SB Attention) from paper:
<Shift-and-Balance Attention> --> https://arxiv.org/abs/2103.13080

Created by Kunhong Yu
Date: 2021/03/28
"""
import torch as t
from torch.nn import functional as F

def weights_init(layer):
	"""
	weights initialization
	Args :
		--layer: one layer instance
	"""
	if isinstance(layer, t.nn.Linear) or isinstance(layer, t.nn.BatchNorm1d):
		t.nn.init.normal_(layer.weight, 0.0, 0.02) # we use 0.02 as initial value
		t.nn.init.constant_(layer.bias, 0.0)

class SBAttention(t.nn.Module):
	"""Define SB operation"""
	"""According to the paper,
		y = T(x) + lambda * A(x)
		where T(x) is the trunk and A(x) = Tanh(F(x))
		F(x) is one simple DNN like in SE, lambda is learnable parameter
		to trade off between T(x) and A(x).
		Notice lambda is channle-wise parameter, so it is one vector.
	"""

	def __init__(self, num_channels, device, attn_ratio, activation = 'tanh'):
		"""
        Args :
		 --num_channels: # of input channels
			 --device: learning device
			 --attn_ratio: hidden size ratio
			 --activation: 'tanh' as default
		"""
		super(SBAttention, self).__init__()

		self.num_channels = num_channels
		self.hidden_size = int(attn_ratio * self.num_channels)

		# 1. Trunk, we use T(x) = x like in SE
		# 2. SB attention
		if activation == 'tanh':
			ac = t.nn.Tanh()

		elif activation == 'sigmoid':
			ac = t.nn.Sigmoid()

		elif activation == 'relu':
			ac = t.nn.ReLU(inplace = True)

		elif activation == 'softmax':
			ac = t.nn.Softmax(dim = -1)

		elif activation == 'linear':
			ac = t.nn.Identity()

		else:
			raise Exception('No other activations!')

		self.lambd = t.nn.Parameter(t.randn(1, self.num_channels, device = device), requires_grad = True)
		self.SB = t.nn.Sequential(
			t.nn.Linear(self.num_channels, self.hidden_size),
			t.nn.BatchNorm1d(self.hidden_size),
			t.nn.ReLU(inplace = True),

			t.nn.Linear(self.hidden_size, self.num_channels),
			t.nn.BatchNorm1d(self.num_channels),
			ac
		)

	def forward(self, x):
		# 1. T(x)
		Tx = x
		# 2. SB attention
		x = F.adaptive_avg_pool2d(x, (1, 1)) # global average pooling
		x = x.squeeze()
		Ax = self.SB(x)

		# 3. output
		x = Tx + t.unsqueeze(t.unsqueeze(self.lambd * Ax, dim = -1), dim = -1) # broadcasting

		return x

# unit test
if __name__ == '__main__':
	sb = SBAttention(1024, t.device('cpu'), 0.5)
	print(sb)
