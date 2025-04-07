import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

"""
LeNet强化学习模型
===============

核心特点：
----------
1. 基于LeNet的深度Q网络
2. 经验回放机制
3. 目标网络更新
4. 状态-动作值函数估计
5. 策略优化能力

实现原理：
----------
1. 使用LeNet进行特征提取
2. 实现Q-learning算法
3. 经验回放缓冲区
4. 目标网络固定
5. 探索与利用平衡

评估指标：
----------
1. 累积奖励
2. 平均Q值
3. 探索率
4. 训练损失
5. 策略稳定性
"""

class LeNetDQN(nn.Module):
    """基于LeNet的深度Q网络"""
    def __init__(self, n_actions):
        super(LeNetDQN, self).__init__()
        
        # 特征提取网络
        self.features = nn.Sequential(
            # 3x32x32 -> 4x28x28
            nn.Conv2d(3, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x14x14
            
            # 4x14x14 -> 8x10x10
            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8x5x5
        )
        
        # 价值估计网络
        self.value = nn.Sequential(
            nn.Linear(8 * 5 * 5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_actions)
        )
        
        # 评估指标
        self.metrics = {
            'rewards': [],
            'q_values': [],
            'losses': [],
            'exploration_rate': []
        }
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        q_values = self.value(x)
        return q_values

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class LeNetRLAgent:
    """LeNet强化学习智能体"""
    def __init__(self, n_actions, device='cuda'):
        self.device = device
        self.n_actions = n_actions
        
        # 创建网络
        self.policy_net = LeNetDQN(n_actions).to(device)
        self.target_net = LeNetDQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(10000)
        
        # 超参数
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update = 10
    
    def select_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()
    
    def optimize_model(self):
        """优化模型"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样经验
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # 转换为张量
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # 更新评估指标
        self.policy_net.metrics['losses'].append(loss.item())
        self.policy_net.metrics['exploration_rate'].append(self.epsilon)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_metrics(self, reward, q_value):
        """更新评估指标"""
        self.policy_net.metrics['rewards'].append(reward)
        self.policy_net.metrics['q_values'].append(q_value)
    
    def print_metrics(self):
        """打印评估指标"""
        print("\n模型评估指标:")
        print(f"平均奖励: {np.mean(self.policy_net.metrics['rewards']):.4f}")
        print(f"平均Q值: {np.mean(self.policy_net.metrics['q_values']):.4f}")
        print(f"平均损失: {np.mean(self.policy_net.metrics['losses']):.4f}")
        print(f"当前探索率: {self.epsilon:.4f}")
    
    def visualize_training(self):
        """可视化训练过程"""
        plt.figure(figsize=(15, 10))
        
        # 绘制奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.policy_net.metrics['rewards'])
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 绘制Q值曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.policy_net.metrics['q_values'])
        plt.title('Q Values')
        plt.xlabel('Step')
        plt.ylabel('Q Value')
        
        # 绘制损失曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.policy_net.metrics['losses'])
        plt.title('Losses')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        
        # 绘制探索率曲线
        plt.subplot(2, 2, 4)
        plt.plot(self.policy_net.metrics['exploration_rate'])
        plt.title('Exploration Rate')
        plt.xlabel('Step')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.show()

def test():
    """测试函数"""
    # 创建智能体
    agent = LeNetRLAgent(n_actions=4)
    
    # 创建随机状态
    state = np.random.rand(3, 32, 32)
    next_state = np.random.rand(3, 32, 32)
    
    # 选择动作
    action = agent.select_action(state)
    print(f"选择的动作: {action}")
    
    # 存储经验
    agent.memory.push(state, action, 1.0, next_state, False)
    
    # 优化模型
    agent.optimize_model()
    
    # 更新评估指标
    agent.update_metrics(1.0, 0.5)
    agent.print_metrics()
    
    # 可视化训练过程
    agent.visualize_training()

if __name__ == '__main__':
    test() 