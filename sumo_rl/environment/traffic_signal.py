import os
import sys
from typing import Callable, List, Union

from sumo_rl.util.normalization import normalize_reward

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gym import spaces


class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """

    def __init__(self,
                 env,
                 ts_id: List[str],
                 delta_time: int,
                 yellow_time: int,
                 min_green: int,
                 max_green: int,
                 begin_time: int,
                 reward_fn: Union[str, Callable],
                 reward_norm_ranges: dict,
                 observation_fn: str,
                 observation_c: float,
                 sumo,
                 ):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.reward_fn = reward_fn
        self.reward_norm_ranges = reward_norm_ranges
        self.observation_fn = observation_fn
        self.observation_c = observation_c
        self.sumo = sumo

        self.build_phases()

        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}

        self.observation_space, self.n_feats = self.get_observation_space()
        self.action_space = spaces.Discrete(self.num_green_phases)

    def build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases)//2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, new_phase):
        """
        Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        :param new_phase: (int) Number between [0..num_green_phases] 
        """
        new_phase = int(new_phase)
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + self.delta_time
        else:
            #self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time
            self.is_yellow = True
            self.time_since_last_phase_change = 0
    
    def compute_observation(self):
        if self.observation_fn == 'dtse':
            return self._dtse_observation()
        elif self.observation_fn == 'density-queue':
            return self._density_queue_observation()
        else:
            raise NotImplementedError(f'Observation function {self.observation_fn} not implemented')

    def compute_reward(self):
        if type(self.reward_fn) is str:
            if self.reward_fn == 'diff_waiting_time':
                self.last_reward = self._diff_waiting_time_reward()
            if self.reward_fn == 'wait':
                self.last_reward = self._wait_reward()
            elif self.reward_fn == 'average_speed':
                self.last_reward = self._average_speed_reward()
            elif self.reward_fn == 'queue':
                self.last_reward = self._queue_reward()
            elif self.reward_fn == 'pressure':
                self.last_reward = self._pressure_reward()
            elif self.reward_fn == 'brake':
                self.last_reward = self._brake_reward()
            elif self.reward_fn == 'emission':
                self.last_reward = self._emission_reward()
            else:
                raise NotImplementedError(f'Reward function {self.reward_fn} not implemented')
            if any(self.reward_norm_ranges.values()):
                self.last_reward = normalize_reward(self.last_reward,
                                                    sample_range=self.reward_norm_ranges[self.reward_fn])
            return self.last_reward
        else:
            return self.reward_fn(self)

    def _density_queue_observation(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + density + queue, dtype=np.float32)
        return observation

    def _dtse_observation(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        observation = np.array([phase_id], dtype=np.float32)
        for lane in self.lanes:
            vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
            positions = (np.array(
                [self.sumo.vehicle.getLanePosition(veh) for veh in vehs]) / self.observation_c).astype(int)
            speeds = np.array([self.sumo.vehicle.getSpeed(veh) for veh in vehs])

            # generate the dtse arrays from position and speed
            dtse_counts, _ = np.histogram(positions, bins=range(self.n_feats + 1))
            dtse_speeds = np.zeros(self.n_feats, dtype=np.float32)
            for speed, pos in zip(speeds, positions):
                dtse_speeds[pos] += (speed / dtse_counts[pos])

            # normalize and clip speeds and occupancies
            normalized_speeds = np.clip(dtse_speeds / self.sumo.lane.getMaxSpeed(lane), 0, 1)
            densities = np.clip(dtse_counts.astype(np.float32) / (self.observation_c / 7.5), 0, 1)

            observation = np.append(observation, normalized_speeds)
            observation = np.append(observation, densities)
        return observation

    def _pressure_reward(self):
        return -self.get_pressure()
    
    def _average_speed_reward(self):
        return self.get_average_speed() - 1

    def _queue_reward(self):
        return -self.get_total_queued()

    def _brake_reward(self):
        return self.get_total_braking()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane())
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _wait_reward(self):
        return - np.mean(self.get_waiting_time_per_lane())

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def _emission_reward(self):
        return - self.get_total_emission()

    def get_observation_space(self):
        if self.observation_fn == 'dtse':
            # all lanes must have equal length for DTSE features
            assert len(set(list(self.lanes_length.values()))) == 1
            lane_length = list(self.lanes_length.values())[0]
            n_feats = int(np.ceil(lane_length / self.observation_c))
        elif self.observation_fn == 'density-queue':
            n_feats = 1
        else:
            raise NotImplementedError(f'Observation function {self.observation_fn} not implemented')

        observation_space = spaces.Box(
            low=np.zeros(self.num_green_phases + 2 * n_feats * len(self.lanes), dtype=np.float32),
            high=np.ones(self.num_green_phases + 2 * n_feats * len(self.lanes), dtype=np.float32))

        return observation_space, n_feats

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self):
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_length[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_total_emission(self):
        return sum(self.sumo.vehicle.getCO2Emission(veh) for veh in self._get_veh_list())

    def get_total_queued(self):
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def get_total_braking(self):
        accelerations = np.array([self.sumo.vehicle.getAcceleration(v) for v in self._get_veh_list()])
        brake = np.sum(accelerations[accelerations < 0])
        return brake

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
