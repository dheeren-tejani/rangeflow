from .layers import RangeLinear, RangeReLU, RangeModule
from .core import RangeTensor
import numpy as np
import torch
import torch.nn as nn

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

    def select_action(self, state, valid_mask=None, epsilon=0.0, mode='robust', device='cpu'):
        """
        Selects action based on Q-value ranges.
        
        Args:
            state: Current state vector
            valid_mask: Binary mask of valid actions (1=valid, 0=invalid)
            epsilon: Exploration rate
            mode: 'robust' (MaxMin) or 'standard' (MaxAvg)
        """
        # 1. Exploration
        if np.random.rand() < epsilon:
            if valid_mask is not None:
                valid_indices = np.where(valid_mask == 1)[0]
                return np.random.choice(valid_indices)
            return np.random.randint(0, self.model.layers[-1].weight.shape[0])
            
        # 2. RangeFlow Inference
        # Wrap state in epsilon ball to model uncertainty (e.g., hidden cards)
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=device)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Create uncertainty interval around state
        # This represents "I am mostly sure, but small details might be wrong"
        x_range = RangeTensor.from_epsilon_ball(state, 0.1)
        
        q_range = self.forward(x_range)
        min_q, max_q = q_range.decay()
        
        # 3. Action Masking (Crucial for logic games)
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                valid_mask = torch.tensor(valid_mask, device=device)
            
            # Set invalid moves to -Infinity
            min_q = min_q.clone()
            max_q = max_q.clone()
            min_q[0, valid_mask == 0] = -float('inf')
            max_q[0, valid_mask == 0] = -float('inf')
        
        # 4. Selection Strategy
        if mode == 'robust':
            # Maximin: Pick action with best WORST-CASE outcome
            return torch.argmax(min_q).item()
        elif mode == 'optimistic':
            # Maximax: Pick action with best BEST-CASE outcome
            return torch.argmax(max_q).item()
        else:
            # Standard: Pick action with best AVERAGE outcome
            return torch.argmax((min_q + max_q)/2).item()

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
    
class RangeReLU(RangeModule):
    def forward(self, x):
        return x.relu()
    
class RangePPO:
    """Proximal Policy Optimization with uncertainty"""
    def __init__(self, state_dim, action_dim):
        from .layers import RangeLinear
        self.actor = RangeSequential(
            RangeLinear(state_dim, 64),
            RangeReLU(),
            RangeLinear(64, action_dim)
        )
        self.critic = RangeSequential(
            RangeLinear(state_dim, 64),
            RangeReLU(),
            RangeLinear(64, 1)
        )
    
    def select_action(self, state, uncertainty=0.05):
        state_range = RangeTensor.from_epsilon_ball(state, uncertainty)
        action_logits = self.actor(state_range)
        # Use center for action selection
        return action_logits.avg()
    
def train_dqn_robust(env, agent, episodes, epsilon, learning_rate=0.001):
    """Complete DQN training loop"""
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_safe_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition and update
            # ... (full implementation would include replay buffer)
            
            state = next_state
            total_reward += reward
        
        print(f"Episode {episode}: Reward = {total_reward}")