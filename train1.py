import os
import sys
import optparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as function
import torch.nn as nn
import matplotlib.pyplot as plt

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary
import traci

def get_vehicle_numbers(lanes):
    vehicle_per_lane = dict()
    for each_lane in lanes:
        vehicle_per_lane[each_lane] = 0
        for k in traci.lane.getLastStepVehicleIDs(each_lane):
            if traci.vehicle.getLanePosition(k) > 10:
                vehicle_per_lane[each_lane] += 1
    return vehicle_per_lane

def get_waiting_time(lanes):
    waiting_time = 0
    for lane in lanes:
        waiting_time += traci.lane.getWaitingTime(lane)
    return waiting_time

def phase_duration(junction, phase_time, phase_state):
    traci.trafficlight.setRedYellowGreenState(junction, phase_state)
    traci.trafficlight.setPhaseDuration(junction, phase_time)

class Model(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dims, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, n_actions)
        self.dropout = nn.Dropout(0.2)
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = function.relu(self.linear1(state))
        x = self.dropout(x)
        x = function.relu(self.linear2(x))
        x = self.dropout(x)
        return self.linear3(x)

class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, fc1_dims, fc2_dims, batch_size, n_actions,
                 junctions, max_memory_size=100000, epsilon_dec=0.00025, epsilon_end=0.05):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.junctions = junctions
        self.max_mem = max_memory_size
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.replace_target = 100
        self.iter_cntr = 0
        self.ratio = [0, 0]
        self.loss_per_epoch = []

        self.Q_eval = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target = Model(lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.memory = dict()
        for junction in junctions:
            self.memory[junction] = {
                "state_memory": np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "new_state_memory": np.zeros((max_memory_size, input_dims), dtype=np.float32),
                "reward_memory": np.zeros(max_memory_size, dtype=np.float32),
                "action_memory": np.zeros(max_memory_size, dtype=np.int32),
                "terminal_memory": np.zeros(max_memory_size, dtype=np.bool_),
                "mem_cntr": 0,
            }

    def store_transition(self, state, new_state, action, reward, done, junction):
        index = self.memory[junction]["mem_cntr"] % self.max_mem
        self.memory[junction]["state_memory"][index] = state
        self.memory[junction]["new_state_memory"][index] = new_state
        self.memory[junction]['reward_memory'][index] = reward
        self.memory[junction]['terminal_memory'][index] = done
        self.memory[junction]["action_memory"][index] = action
        self.memory[junction]["mem_cntr"] += 1

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.Q_eval.device)
        if np.random.random() > self.epsilon:
            action = torch.argmax(self.Q_eval(state)).item()
            self.ratio[0] += 1
        else:
            action = np.random.choice(self.action_space)
            self.ratio[1] += 1
        return action

    def save(self, model):
        torch.save(self.Q_eval.state_dict(), f'models/{model}.bin')

    def learn(self, junction):
        if self.memory[junction]['mem_cntr'] < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = self.memory[junction]['mem_cntr']
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.memory[junction]['state_memory'][batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.memory[junction]['new_state_memory'][batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.memory[junction]['reward_memory'][batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.memory[junction]['terminal_memory'][batch]).to(self.Q_eval.device)
        action_batch = self.memory[junction]['action_memory'][batch]

        q_eval = self.Q_eval(state_batch)[np.arange(self.batch_size), action_batch]
        q_next = self.Q_target(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_eval, q_target).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.loss_per_epoch.append(loss.item())
        self.iter_cntr += 1

        if self.iter_cntr % self.replace_target == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_end

def run(train=True, model_name="models", epochs=50, steps=500):
    best_time = np.inf
    total_time_list = []
    epsilon_list = []
    exploitation_list = []
    exploration_list = []

    traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"])
    all_junctions = traci.trafficlight.getIDList()
    junction_numbers = list(range(len(all_junctions)))
    brain = Agent(0.99, 1.0, 0.001, 4, 256, 256, 64, 4, junction_numbers)

    if not train:
        brain.Q_eval.load_state_dict(torch.load(f'models/{model_name}.bin', map_location=brain.Q_eval.device))

    traci.close()

    for e in range(epochs):
        traci.start([checkBinary("sumo"), "-c", "configuration.sumocfg", "--tripinfo-output", "maps/tripinfo.xml"] if train else [checkBinary("sumo-gui"), "-c", "configuration.sumocfg"])
        print(f"Epoch: {e} | Exploring: {brain.ratio[1]} | Exploiting: {brain.ratio[0]} | Epsilon: {brain.epsilon}")

        select_lane = [["yyyrrrrrrrrr", "GGGrrrrrrrrr"], ["rrryyyrrrrrr", "rrrGGGrrrrrr"], ["rrrrrryyyrrr", "rrrrrrGGGrrr"], ["rrrrrrrrryyy", "rrrrrrrrrGGG"]]
        step, total_time, min_duration = 0, 0, 5
        traffic_lights_time, prev_wait_time, prev_vehicles_per_lane, prev_action = {}, {}, {}, {}

        for junction_number, junction in enumerate(all_junctions):
            prev_wait_time[junction] = 0
            prev_action[junction_number] = 0
            traffic_lights_time[junction] = 0
            prev_vehicles_per_lane[junction_number] = [0] * 4

        while step <= steps:
            traci.simulationStep()
            for junction_number, junction in enumerate(all_junctions):
                controlled_lanes = traci.trafficlight.getControlledLanes(junction)
                waiting_time = get_waiting_time(controlled_lanes)
                total_time += waiting_time

                if traffic_lights_time[junction] == 0:
                    vehicles_per_lane = get_vehicle_numbers(controlled_lanes)
                    reward = -1 * waiting_time
                    state_ = list(vehicles_per_lane.values())
                    state = prev_vehicles_per_lane[junction_number]
                    prev_vehicles_per_lane[junction_number] = state_
                    brain.store_transition(state, state_, prev_action[junction_number], reward, (step == steps), junction_number)
                    lane = brain.choose_action(state_)
                    prev_action[junction_number] = lane
                    phase_duration(junction, 6, select_lane[lane][0])
                    phase_duration(junction, min_duration + 10, select_lane[lane][1])
                    traffic_lights_time[junction] = min_duration + 10
                    if train:
                        brain.learn(junction_number)
                else:
                    traffic_lights_time[junction] -= 1
            step += 1

        print("Total waiting time:", total_time)
        total_time_list.append(total_time)
        epsilon_list.append(brain.epsilon)
        exploitation_list.append(brain.ratio[0])
        exploration_list.append(brain.ratio[1])

        if total_time < best_time and train:
            best_time = total_time
            brain.save(model_name)

        traci.close()
        sys.stdout.flush()
        if not train:
            break

    if train:
        os.makedirs("plots", exist_ok=True)

        plt.figure()
        plt.plot(total_time_list)
        plt.xlabel("Epochs")
        plt.ylabel("Total Waiting Time")
        plt.title("Total Time vs Epoch")
        plt.savefig(f'plots/total_time_{model_name}.png')

        plt.figure()
        plt.plot(brain.loss_per_epoch)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.savefig(f'plots/loss_vs_epoch_{model_name}.png')

        plt.figure()
        plt.plot(epsilon_list, label="Epsilon")
        plt.xlabel("Epochs")
        plt.ylabel("Epsilon")
        plt.title("Epsilon Decay Over Time")
        plt.legend()
        plt.savefig(f'plots/epsilon_decay_{model_name}.png')

        plt.figure()
        plt.plot(exploration_list, label="Explore")
        plt.plot(exploitation_list, label="Exploit")
        plt.xlabel("Epochs")
        plt.ylabel("Actions Taken")
        plt.title("Exploration vs Exploitation")
        plt.legend()
        plt.savefig(f'plots/explore_exploit_{model_name}.png')
        plt.show()

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("-m", dest='model_name', type='string', default="model", help="name of models")
    opt_parser.add_option("--train", action='store_true', default=False, help="training or testing")
    opt_parser.add_option("-e", dest='epochs', type='int', default=50, help="Number of epochs")
    opt_parser.add_option("-s", dest='steps', type='int', default=500, help="Number of steps")
    return opt_parser.parse_args()[0]

if __name__ == "__main__":
    options = get_options()
    run(train=options.train, model_name=options.model_name, epochs=options.epochs, steps=options.steps)
