import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)

class Agent(object):
    """ Agent class for use with separate actor and critic networks.
        This is appropriate for very simple environments, such as the mountaincar
    """
    def __init__(self, alpha, beta, input_dims, gamma=0.99,
                 layer1_size=256, layer2_size=256, n_actions=2, action_space='discrete'):
        self.gamma = gamma
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                     layer2_size, n_actions=1)
        self.log_probs = None
        self.action_space = action_space

    def choose_action(self, observation):
        if self.action_space == 'discrete':
            probabilities = F.softmax(self.actor.forward(observation))
            action_probs = T.distributions.Categorical(probabilities)
            action = action_probs.sample()
            self.log_probs = action_probs.log_prob(action)

        elif self.action_space == 'continuous':
            mu, sigma  = self.actor.forward(observation)#.to(self.actor.device)
            sigma = T.exp(sigma)
            action_probs = T.distributions.Normal(mu, sigma)
            probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
            self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
            action = T.tanh(probs)
        else:
            raise NotImplementedError
        return action.item()


    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()