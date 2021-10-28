import pandas as pd
import re
import numpy as np
import shelve
randomizer = np.random.default_rng()


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
        self.state_k = 4  # number of parameters passed saved per state
        self.prob_acc = {}  # list of dictionaries with probability of accident occuring for each region, per zip code
        self.curr_state = []  # saves current state of environment; KxN matrix that we then pass to the environment
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

        environment_data = shelve.open('environment.txt')
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
        :return: total travel time in s
        """
        min_dist_time = 9999
        hospital_loc = None

        for i, hospital in enumerate(self.hospitals[region_nr]):
            dist_time = self.distance_time(region_nr, accident_loc, hospital)
            if dist_time < min_dist_time:
                min_dist_time = dist_time
                hospital_loc = hospital

        initial_time = 1 * 60  # 1 min
        buffer_time = 15 * 60  # 15 mins
        total_travel_time = initial_time + self.distance_time(region_nr, ambulance_loc,
                                                              accident_loc) + buffer_time + self.distance_time(
            region_nr,
            accident_loc,
            hospital_loc) + self.distance_time(region_nr,
                                               hospital_loc, ambulance_loc)

        return total_travel_time

    def create_accidents(self, region_nr):
        """
        Creates a dictionary where keys are the exact second that an accident take place, and value is the zip_code index
        :param region_nr:
        :return:
        """
        accidents_dict = {}
        accidents_per_day = self.accidents[region_nr]/365
        random_values = randomizer.random(86400)
        total_population = sum(self.pop_dic[region_nr])
        percentage_population = []
        for pop_zipcode in self.pop_dic[region_nr]:
            percentage_population.append(float(pop_zipcode) / total_population)
        #86400 is the number of seconds in a day, because a day is one episode's length

        accidents_per_second = accidents_per_day/86400

        for second in range(86400):
            if random_values[second] < accidents_per_second:
                zip_code_index = np.random.choice(len(percentage_population), 1, p=percentage_population)[0]
                accidents_dict.update({second: zip_code_index})

        return accidents_dict
