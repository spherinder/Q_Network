import gym
import tensorflow as tf
import numpy as np

env = gym.make("FrozenLake-v0")

x = tf.placeholder(tf.float32, [1, env.observation_space.n])
w = tf.Variable(tf.random_normal([16, 4], mean=0.005,stddev=0.01))
out = tf.matmul(x, w)

target = tf.placeholder(tf.float32, [1, env.action_space.n])
error = tf.reduce_sum(tf.squared_difference(target, out))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(error)

saver = tf.train.Saver()

GAMMA = .99
epsilon = .1
rewards = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./tmp/DQN.ckpt")
    for i in range(2000):
        state = env.reset()
        done = False
        episode_reward = 0
        for j in range(100):
            # GET INITIAL Q-VALUES
            all_Q = sess.run(out, {
                x: np.identity(16)[state:state+1]
            })
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(all_Q)

            # TAKE STEP AND CALCULATE BETTER Q-VALUES
            next_s, reward, done, _ = env.step(action)
            targetQ = all_Q
            targetQ[0, action] = reward + GAMMA * np.max(sess.run(out, {
                x: np.identity(16)[next_s:next_s+1]}))

            # UPDATE
            sess.run(optimizer, {
                x: np.identity(16)[state:state+1], target: targetQ})

            episode_reward += reward
            if done:
                epsilon = 1. / (i / 50 + 10)
                break
            state = next_s
        rewards.append(episode_reward)
    saver.save(sess, "./tmp/DQN.ckpt")
if __name__ == "__main__":
    print(rewards)
