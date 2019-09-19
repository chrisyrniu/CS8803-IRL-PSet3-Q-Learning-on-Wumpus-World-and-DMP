from wumpus_world_new import *
import matplotlib.pyplot as plt

max_episode = 10000
episode_duration = 16
episode = 1
gamma = 0.95
epsilon = 0.1
alpha = 0.01
q_table = dict()
loss_list = []
states = set()
state_number = []
iterations = 0

while episode <= max_episode:
	dataset = []
	time = 1
	while time < 16:
		data = []
		if time == 1:
			agent = Agent((1,1), (4,4), 0.9)
			wumpus = Wumpus((1,3), (4,4))
			world = MDP((4,4), (2,3), [(3,1), (3,3), (4,4)], wumpus, agent, (1,1))
			state = world.init_state()
			states.add(state)
			data.append(state)
			reward = world.get_reward()
			if world.is_terminal() == True:
				data.append(reward)
				dataset.append(data)			
				break
			q_table = world.expand_q_table(q_table, state)
			exploit_actions = agent.choose_exploit_action(state, q_table)
			action = agent.epsilon_greedy(exploit_actions, epsilon)
			data.append(action)
			data.append(reward)
			wumpus.move_randomly()
			state = world.update_state(state)
			states.add(state)
			data.append(state)
		else:
			data.append(state)
			reward = world.get_reward()
			if world.is_terminal() == True:
				data.append(reward)
				dataset.append(data)			
				break
			q_table = world.expand_q_table(q_table, state)
			exploit_actions = agent.choose_exploit_action(state, q_table)
			action = agent.epsilon_greedy(exploit_actions, epsilon)
			data.append(action)
			data.append(reward)
			wumpus.move_randomly()
			state = world.update_state(state)
			states.add(state)
			data.append(state)
			if time == 15:
				q_table = world.expand_q_table(q_table, state)
		dataset.append(data)
		time = time + 1
		state_number.append(len(states))
		iterations = iterations + 1
	loss = 0
	for i in range(len(dataset) - 1, -1, -1):
		if len(dataset[i]) < 4:
			current_state = dataset[i][0]
			current_reward = dataset[i][1]
			if ((current_state, "stay") in q_table) == False:
				q_table[(current_state, "stay")] = 0
			q_table[(current_state, "stay")] = (1 - alpha) * q_table[(current_state, "stay")] + alpha * current_reward
			loss = loss + (current_reward - q_table[(current_state, "stay")])**2
		else:
			current_state = dataset[i][0]
			current_action = dataset[i][1]
			current_reward = dataset[i][2]
			next_state = dataset[i][3]
			next_q_values = []
			for key in q_table:
				if key[0] == next_state:
					next_q_values.append(q_table[key])
			max_q_value = max(next_q_values)
			q_table[current_state, current_action] = (1 - alpha) * q_table[current_state, current_action] + alpha * (current_reward + gamma * max_q_value)
			loss = loss + (current_reward + gamma * max_q_value - q_table[current_state, current_action])**2
	loss_list.append(loss)
	episode = episode + 1
agent = Agent((1,1), (4,4), 0.9)
wumpus = Wumpus((1,3), (4,4))
world = MDP((4,4), (2,3), [(3,1), (3,3), (4,4)], wumpus, agent, (1,1))
state = world.init_state()
print(q_table[(state, "up")])
print(q_table[(state, "right")])
print(q_table[(state, "stay")])

plt.scatter(range(1, max_episode + 1), loss_list, s = 5, marker = ".")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.show()
plt.plot(range(1, iterations + 1), state_number)
plt.xlabel("Iteration")
plt.ylabel("Unique State Number")
plt.show()


