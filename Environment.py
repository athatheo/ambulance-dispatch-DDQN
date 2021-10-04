import pandas as pd
import openpyxl
import re
import numpy as np


class Environment:

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
            reg_bases = {l[0]: l[1] for l in temp}
            self.bases.update({i + 1: reg_bases})

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
                accZip = accidents * (float(pop_zipcode) / totPop) # is this correct???

            self.prob_acc.update({region_nr: accZip})

    def distance_time(self, a, b):
        """
        Caluclates the travel time between two postal codes
        :param a: starting point of the measured time
        :param b: ending point of the measured time
        :return: travel time in s
        """
        region = 0
        for i in self.postcode_dic:
            if (a in self.postcode_dic[i]):
                region = i
                break
            elif (b in self.postcode_dic[i]):
                region = i
                break

        A = self.postcode_dic[region].index(a)
        B = self.postcode_dic[region].index(b)

        if region < 13:
            i = region - 1
        else:
            i = region - 2
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
        res = initial_time + self.distance_time(ambulance_loc, accident_loc) + buffer_time + self.distance_time(
            accident_loc,
            hospital_loc) + self.distance_time(
            hospital_loc, ambulance_loc)

        return res

    def sample_accidents(self, region_nr):
        """
        sample accidents uniformly over time
        :param: region number for region of interest
        :return: list of booleans indicating if accident happened or not per zipcode
        """
        accidentsYear = self.accidents[region_nr]
        totPop = sum(self.pop_dic[region_nr])  # get total number of people for region number
        accidents = accidentsYear / 365  # per day
        accidents = accidents / 86400  # per seconds

        per = []
        accZip = []
        bool_acc = []
        for i in self.pop_dic[region_nr]:
            a = float(i) / totPop
            per.append(a)  # Percentage of people per zipcode (people/total people)
            accZip.append(accidents * i)  # percentage of accidents per zipcode
            # sample boolean vector
            if np.random.rand() <= accidents* a:
                bool = 1
                bool_acc.append(bool)
            else:
                bool = 0
                bool_acc.append(bool)

        return bool_acc

    def initialize_space(self, region_nr):
        """
        At the beginning of an episode initialize a state for the given region with all available ambulances.
        Initializes it with time of day = 0 and travel time = 0 because no accidents have occured yet.
        :param region_nr: number of region from current episode
        """

        N = len(self.postcode_dic[region_nr])
        self.curr_state = []
        # initialize K-array for each zip code: travel time, delta, time of day
        for i in range(N):
            curr_zipcode = self.postcode_dic[region_nr][i]
            if curr_zipcode in self.bases[region_nr]:
                is_base = 1
            else:
                is_base = 0

            self.curr_state.append([0, #boolean accident = 0
                                    self.nr_ambulances[region_nr], # nr ambulances
                                    is_base, # boolean base
                                    0, # travel time = 0 because no accident
                                    self.prob_acc[region_nr][i], # delta = probability of next accident occuring in this zipcode
                                    0]) # time of day

        self.k = len(self.curr_state[-1])  # number of parameters available

    def process_action(self, action):
        """Takes an action (ambulance sent out) and returns the new state and reward."""
        return None