import pandas as pd
import openpyxl
import re
import numpy as np
import torch
import shelve

class Environment:
    """Environment class containing information about all ambulances, regions, zipcodes, populations and hospitals."""

    def __init__(self):

        self.pop_dic = {}  # for each region the population per postcode
        self.postcode_dic = {}  # for each region all available postcodes
        self.coverage_lst = []  # list of dictionaries; each dictionary contains the coverage time of one region
        self.accidents = {}  # number of total accidents per region
        self.bases = {}  # dictionary with all ambulance bases for each region
        self.hospitals = {}  # dictionary with region as keys and hospital postcodes as values
        self.nr_postcodes = {}  # records number of postcodes per region
        self.nr_ambulances = {}  # records number of ambulances per region
        self.state_k = 6 # number of parameters passed saved per state
        self.prob_acc = {} # list of dictionaries with probability of accident occuring for each region, per zip code
        self.curr_state = [] # saves current state of environment; KxN matrix that we then pass to the environment

        print("Initialisation complete")

    def import_data(self):
        """
        Imports data for all 25 regions
        :param self:
        :return: postcode_dict: dictionary with the region numbers as key and values corresponding to the postcodes per region
        """

        for i in range(1, 25):
            # skip nonexisting region number
            if i == 13:
                continue

            # read in coverage data (distances between postcodes of one region)
            coverage_df = pd.read_excel('Data/coverage{}.xlsx'.format(i), header=None)

            self.coverage_lst.append(coverage_df.to_dict('list'))

            with open('Data/population{}.txt'.format(i)) as f:
                lines = f.readlines()

            postcode_lst = []
            pop_lst = []

            for line in lines[3:]:
                temp = re.findall(r'\d+', line)
                values = list(map(int, temp))
                postcode_lst.append(values[0])
                pop_lst.append(values[1])

            self.postcode_dic.update({i: postcode_lst})
            self.pop_dic.update({i: pop_lst})

        # number of registered accidents per region
        with open('Data/DataAllRegions.txt') as f:
            lines = f.readlines()

        for line in lines[1:]:
            temp = re.findall(r'\d+', line)
            values = list(map(int, temp))
            self.accidents.update({values[0]: values[1]})

        # location of bases and number of ambulances per base
        with open('Data/xMEXCLP_all.txt') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            res = re.split(',|;', line)
            temp = [re.findall(r'\d+', s) for s in res if re.findall(r'\d+', s) != []]
            reg_bases = {int(l[0]): int(l[1]) for l in temp}
            if i < 13:
                self.bases.update({int(i + 1): reg_bases})
            else:
                self.bases.update({int(i + 2): reg_bases})

        # retrieve the postcode locations for each hospital
        # retrieve the postcode locations for each hospital
        with open('Data/Hospitals.txt') as f:
            lines = f.readlines()

        for line in lines:
            res = re.split(',|:', line.strip('\n'))
            self.hospitals.update({int(res[0]): list(map(int, res[1:]))})

        # extract region sizes and ambulances per region
        ambuRegion = pd.read_excel('Data/NumberPCAmbuRegion.xlsx')
        ambuRegion = ambuRegion.set_index('Region')
        self.nr_postcodes = ambuRegion['Postal codes'].to_dict()
        self.nr_ambulances = ambuRegion['ambulances'].to_dict()
        print("Import data complete")

        # calculate probability of accidents per region and zipcode
        for region_nr in self.postcode_dic:
            accidentsYear = self.accidents[region_nr]
            totPop = sum(self.pop_dic[region_nr])  # get total number of people for region number
            accidents = accidentsYear / 365  # per day
            accidents = accidents / 86400  # per seconds

            accZip = []
            for pop_zipcode in self.pop_dic[region_nr]:
                accZip.append(accidents * (float(pop_zipcode) / totPop))

            self.prob_acc.update({region_nr: accZip})

        environment_data = shelve.open('environment.db')
        environment_data['key'] = self

    def distance_time(self, region_nr, a, b):
        """
        Caluclates the travel time between two postal codes
        :param a: starting point of the measured time
        :param b: ending point of the measured time
        :return: travel time in s
        """

        A = self.postcode_dic[region_nr].index(a)
        B = self.postcode_dic[region_nr].index(b)

        if region_nr < 13:
            i = region_nr - 1
        else:
            i = region_nr - 2
        return self.coverage_lst[i][B][A]

    def calculate_ttt(self, region_nr, ambulance_loc, accident_loc):
        """
        Caluclates the total travel time for an ambulance to the CLOSEST hosptial plus 15 min buffer
        :param ambulance_loc: postal code of the ambulance location
        :param accident_loc: postal code of the accident location
        :param hospital_loc: postal code of the hospital location
        :return: total travel time in s
        """
        min_dist_time = 9999
        hospital_loc = None

        for i, hospital in enumerate(self.hospitals[region_nr]):
            dist_time = self.distance_time(accident_loc, hospital)
            if dist_time < min_dist_time:
                min_dist_time = dist_time
                hospital_loc = hospital

        initial_time = 1 * 60  # 1 min
        buffer_time = 15 * 60  # 15 mins
        res = initial_time + self.distance_time(region_nr, ambulance_loc, accident_loc) + buffer_time + self.distance_time(region_nr,
            accident_loc,
            hospital_loc) + self.distance_time(region_nr,
            hospital_loc, ambulance_loc)

        return res

    def sample_accidents(self, region_nr):
        """
        sample accidents uniformly over time
        :param: region number for region of interest
        :return: list of booleans indicating if accident happened or not per zipcode
        """
        accident_prob = self.prob_acc[region_nr]
        bool_acc = []
        
        # sample boolean vector
        if np.random.rand() <= accident_prob:
            bool = 1
        else:
            bool = 0
        bool_acc.append(bool)

        return bool_acc

class State:

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

    def process_action(self, action, time, accident_loc):
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
            self.nr_ambulances[action] -= 1
            total_travel_time = self.env.calculate_ttt(self.region_nr, self.env.postcode_dic[self.region_nr][action], accident_loc)
            self.ambulance_return.update({total_travel_time + time: action})

        return self, -self.travel_time[action]

    def update_state(self, time, accident_list):
        self.bool_accident = accident_list

        for i in range(len(self.N)):
            self.time[i] = time

        self.update_travel_time(self.get_accident_location(accident_list))

    def get_accident_location(self, booleanList):
        for i in range(len(booleanList)):
            if booleanList[i] == 1:
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