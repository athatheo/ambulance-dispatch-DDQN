import torch
from torch import device, cuda
import numpy as np
device = device("cuda" if cuda.is_available() else "cpu")


class State(object):

    def __init__(self, env, region_nr):
        """"
        New state space at beginning of each episode.
        Initializes a state for the given zipcode in the specified region.
        All ambulances are initially available and no accidents have occured yet.
        """
        # Doesn't it initialize a state for the region and not the specific zipcode?
        self.env = env
        self.K = 4
        self.N = len(env.postcode_dic[region_nr])
        self.ambulance_return = {}  # Dictionary with key: when will an ambulance return and value: zip code of base
        self.region_nr = region_nr
        self.waiting_list = []
        self.accident_location_index = -1

        # parameters initialized to pass to NN
        self.bool_accident = [0] * self.N
        # is_base: boolean list of whether an env.postcode_dic[region_nr][i] zip_code is a base
        # nr_ambulances: int list of how many ambulances are available per zip_code
        self.is_base, self.nr_ambulances = self.check_isBase()
        self.travel_time = [0] * self.N  # time from base to accident
        self.delta = env.prob_acc[region_nr]
        self.time = [0] * self.N

        # index of bases (should not be masked)
        self.indexNotMasked = np.array([i for i, e in enumerate(self.is_base) if e == 1])

    def check_isBase(self):
        """
        Find all bases in a region
        :param env:
        :param region_nr:
        :return: boolean list indicating if zip-code has a base or not
        """
        isBase = []
        nr_ambulances = []
        for zip_code in self.env.postcode_dic[self.region_nr]:
            if zip_code in self.env.bases[self.region_nr]:
                isBase.append(1)
                nr_ambulances.append(self.env.bases[self.region_nr][zip_code])
            else:
                isBase.append(0)
                nr_ambulances.append(0)
        return isBase, nr_ambulances

    def update_travel_time(self, accident_loc):
        """
        Takes the accident location zip code and updates the travel_time list.
        :param accident_loc: zip code that the ambulance is sent to
        """
        for i, zip_code in enumerate(self.env.postcode_dic[self.region_nr]):
            if zip_code in self.env.bases[self.region_nr]:
                if self.nr_ambulances[i] > 0:
                    self.travel_time[i] = self.env.distance_time(self.region_nr, zip_code, accident_loc)
                else:
                    self.travel_time[i] = 99999
            else:
                self.travel_time[i] = 99999

    def update_travel_time_greedy(self, accident_loc):
        """
        Takes the accident location zip code and updates the travel_time list.
        :param accident_loc: zip code that the ambulance is sent to
        """
        for i, zip_code in enumerate(self.env.postcode_dic[self.region_nr]):
            if zip_code in self.env.bases[self.region_nr]:
                if self.nr_ambulances[i] > 0:
                    self.travel_time[i] = self.env.distance_time(self.region_nr, zip_code, accident_loc)
                else:
                    self.travel_time[i] = 99999
            else:
                self.travel_time[i] = 99999

    def process_action(self, action, time):
        """
        Takes an action (ambulance sent out) and returns the new state and reward.
        :param action: index of zip code that send out ambulance
        :param time: time of the day in seconds that ambulance was sent out
        :param accident_loc: location of the accident to calculate when an amublance will arrive
        :return reward: minus time from ambulance to the accident
        """
        if action == 0:

            return -200
        else:
            accident_loc = self.get_accident_location()
            self.nr_ambulances[action] -= 1
            if self.nr_ambulances[action] == 0:
                self.indexNotMasked = self.indexNotMasked[self.indexNotMasked != action]
            total_travel_time = self.env.calculate_ttt(self.region_nr, self.env.postcode_dic[self.region_nr][action],
                                                       accident_loc)
            self.ambulance_return.update({total_travel_time + time: action})
        if self.travel_time[action] == 0:
            return 10000 / 100
        return 10000/self.travel_time[action]

    def process_action_greedy(self, action, time):
        """
        Takes an action (ambulance sent out) and returns the new state and reward.
        :param action: index of zip code that send out ambulance
        :param time: time of the day in seconds that ambulance was sent out
        :param accident_loc: location of the accident to calculate when an amublance will arrive
        :return reward: minus time from ambulance to the accident
        """
        if action == 0 or self.nr_ambulances[action] == 0:

            return -200
        else:
            accident_loc = self.get_accident_location()
            self.nr_ambulances[action] -= 1
            if self.nr_ambulances[action] == 0:
                self.indexNotMasked = self.indexNotMasked[self.indexNotMasked != action]
            total_travel_time = self.env.calculate_ttt(self.region_nr, self.env.postcode_dic[self.region_nr][action],
                                                       accident_loc)
            self.ambulance_return.update({total_travel_time + time: action})
        if self.travel_time[action] == 0:
            return 10000 / 100

        return 10000/self.travel_time[action]

    def update_state(self, time, zip_code_index):
        if self.accident_location_index != -1:
            self.bool_accident[self.accident_location_index] = 0
        self.bool_accident[zip_code_index] = 1
        self.accident_location_index = zip_code_index
        for i in range(self.N):
            self.time[i] = time

        self.update_travel_time(self.get_accident_location())

    def update_state_greedy(self, time, zip_code_index):
        if self.accident_location_index != -1:
            self.bool_accident[self.accident_location_index] = 0
        self.bool_accident[zip_code_index] = 1
        self.accident_location_index = zip_code_index
        for i in range(self.N):
            self.time[i] = time

        self.update_travel_time_greedy(self.get_accident_location())

    def get_accident_location(self):
        return self.env.postcode_dic[self.region_nr][self.accident_location_index]


    def get_torch(self):
        """
        Transform state object into a KxN torch, where K = number of parameters and N = number of zipcodes
        :return:
        """
        return torch.transpose(torch.tensor([self.nr_ambulances,
                                             self.travel_time,
                                             self.time,
                                             self.is_base], device=device), 0, 1)
        return torch.transpose(torch.tensor([self.bool_accident,
                                             self.nr_ambulances,
                                             self.is_base,
                                             self.travel_time,
                                             self.delta,
                                             self.time], device=device), 0, 1)

    def __deepcopy__(self, memodict={}):
        copy_object = State(self.env, self.region_nr)

        copy_object.ambulance_return = self.ambulance_return.copy()
        copy_object.waiting_list = self.waiting_list.copy()

        # parameters initialized to pass to NN
        copy_object.bool_accident = self.bool_accident.copy()
        # is_base: boolean list of whether an env.postcode_dic[region_nr][i] zip_code is a base
        # nr_ambulances: int list of how many ambulances are available per zip_code
        copy_object.is_base, copy_object.nr_ambulances = self.is_base.copy(), self.nr_ambulances.copy()
        copy_object.travel_time = self.travel_time.copy()
        copy_object.delta = self.delta.copy()
        copy_object.time = self.time.copy()

        # index of bases (should not be masked)
        copy_object.indexNotMasked = self.indexNotMasked.copy()
        return copy_object





