import gym
import numpy as np


class TwoLayerReluNN(object):
    def __init__(self, theta):
        self.w0, self.b0, self.w1, self.b1 = theta[:24].reshape(8, 3), theta[24:27], theta[27:30].reshape(3, 1), theta[30:]

    def act(self, ob):
        h = ob.dot(self.w0) + self.b0
        h[h<0] = 0
        y = h.dot(self.w1) + self.b1
        return y


def get_proposals(th_mean, batch_size, std=10):
    stds = np.ones_like(th_mean) * std
    return np.array([th_mean + std for std 
        in stds[None,:] * np.random.randn(batch_size, th_mean.size)])


def cross_entropy(f, curr_mean, batch_size, num_iter, elite_frac):
    n_elite = int(np.round(batch_size * elite_frac))
    for _ in range(num_iter):
        proposals = get_proposals(curr_mean, batch_size, std=5)
        ys = np.array([f(proposal) for proposal in proposals])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_proposals = proposals[elite_inds]
        curr_mean = elite_proposals.mean(axis=0)
        curr_std = elite_proposals.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : curr_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t % 1 == 0: 
            env.render()
        if done: break
    return total_rew, t + 1

if __name__ == '__main__':
    env = gym.make("ReacherSingle-v0")
    env.seed(0)
    np.random.seed(0)

    def eval_twoLayerReluNN(theta):
        agent = TwoLayerReluNN(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # env.observation_space.shape[0] == 11
    params = dict(num_iter=500, batch_size=1000, elite_frac=0.01)
    num_steps = 200
    for (i, iterdata) in enumerate(
        cross_entropy(eval_twoLayerReluNN, np.zeros((8 + 1) * 3 + (3 + 1) * 1), **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        print(iterdata['theta_mean'])
        agent = TwoLayerReluNN(iterdata['theta_mean'])
        do_rollout(agent, env, 200, render=True)

    env.close()




