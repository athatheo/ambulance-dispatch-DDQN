import torch
import Environment


class State(object):

    def __init__(self, env, region_nr):
        """"
        New state space at beginning of each episode.
        Initializes a state for the given zipcode in the specified region.
        All ambulances are initially available and no accidents have occured yet.
        """
        # Doesn't it initialize a state for the region and not the specific zipcode?
        self.env = env
        self.K = 6
        self.N = len(env.postcode_dic[region_nr])
        self.ambulance_return = {}  # Dictionary with key: when will an ambulance return and value: zip code of base
        self.region_nr = region_nr
        self.waiting_list = []

        # parameters initialized to pass to NN
        self.bool_accident = [0] * self.N
        # is_base: boolean list of whether an env.postcode_dic[region_nr][i] zip_code is a base
        # nr_ambulances: int list of how many ambulances are available per zip_code
        self.is_base, self.nr_ambulances = self.check_isBase()
        self.travel_time = [0] * self.N  # time from base to accident
        self.delta = env.prob_acc[region_nr]
        self.time = [0] * self.N

        # index of bases (should not be masked)
        self.indexNotMasked = [i for i, e in enumerate(self.is_base) if e == 1]

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
        Takes the accident location and updates the travel_time list.
        :param accident_loc: zip code that the ambulance is sent to
        """
        for i, zip_code in enumerate(self.env.postcode_dic[self.region_nr]):
            if zip_code in self.env.bases[self.region_nr]:
                self.travel_time[i] = self.env.distance_time(self.region_nr, zip_code, accident_loc)
            else:
                self.travel_time[i] = 0

    def process_action(self, action, time):
        """
        Takes an action (ambulance sent out) and returns the new state and reward.
        :param action: index of zip code that send out ambulance
        :param time: time of the day in seconds that ambulance was sent out
        :param accident_loc: location of the accident to calculate when an amublance will arrive
        :return reward: minus time from ambulance to the accident
        """
        if self.nr_ambulances[action] < 1:
            # We need to add waiting list here
            raise ValueError("No ambulances available to send out.")
        else:
            accident_loc = self.get_accident_location()
            self.nr_ambulances[action] -= 1
            total_travel_time = self.env.calculate_ttt(self.region_nr, self.env.postcode_dic[self.region_nr][action],
                                                       accident_loc)
            self.ambulance_return.update({total_travel_time + time: action})

        return self, -self.travel_time[action]

    def update_state(self, time, accident_list):
        self.bool_accident = accident_list

        for i in range(self.N):
            self.time[i] = time

        self.update_travel_time(self.get_accident_location())

    def get_accident_location(self):

        for i in range(len(self.bool_accident)):
            if self.bool_accident[i] == 1:
                accident_index = i
        return self.env.postcode_dic[self.region_nr][accident_index]

    def get_torch(self):
        """
        Transform state object into a KxN torch, where K = number of parameters and N = number of zipcodes
        :return:
        """
        return torch.tensor([self.bool_accident,
                             self.nr_ambulances,
                             self.is_base,
                             self.travel_time,
                             self.delta,
                             self.time]).transpose(self.K, self.N)
