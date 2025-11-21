from .layers import RangeLinear, RangeRelu
from .core import RangeTensor
import numpy as np

class RangeDQN:
    """
    A Deep Q-Network that estimates Q-value ranges.
    Used for Risk-Averse Reinforcement Learning.
    """
    def __init__(self, state_dim, action_dim):
        self.l1 = RangeLinear(state_dim, 64)
        self.l2 = RangeLinear(64, 64)
        self.l3 = RangeLinear(64, action_dim)
        self.relu = RangeRelu()

    def forward(self, state_range):
        x = self.relu(self.l1(state_range))
        x = self.relu(self.l2(x))
        return self.l3(x)

    def select_safe_action(self, state, uncertainty=0.05):
        """
        Selects action based on Worst-Case Q-value (MaxMin policy).
        Guarantees a baseline reward even in worst noise.
        """
        # 1. Wrap state in range (Sensor Noise)
        state_range = RangeTensor.from_array(state)
        noise = RangeTensor.from_array(np.ones_like(state) * uncertainty)
        robust_state = state_range + noise
        
        # 2. Get Q-Value Ranges
        q_range = self.forward(robust_state)
        min_q, max_q = q_range.decay()
        
        # 3. Pessimistic Selection (Safety First)
        # We pick the action with the highest MINIMUM Q-value.
        # "Which action is best if everything goes wrong?"
        safe_action = np.argmax(min_q)
        return safe_action

    def select_optimistic_action(self, state, uncertainty=0.05):
        """
        Exploration Mode: Optimistic Selection (MaxMax).
        "Which action has the highest potential?"
        """
        state_range = RangeTensor.from_array(state)
        robust_state = state_range + (RangeTensor.from_array(np.ones_like(state) * uncertainty))
        
        q_range = self.forward(robust_state)
        min_q, max_q = q_range.decay()
        
        return np.argmax(max_q)