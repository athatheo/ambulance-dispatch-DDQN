import pandas as pd
import openpyxl
import re

class environment:

    def __init__(self):

        self.pop_dic = {}  # for each region the population per postcode
        self.postcode_dic = {}  # for each region all available postcodes
        self.coverage_lst = []  # list of dictionaries; each dictionary contains the coverage time of one region
        self.accidents = {} # number of total accidents per region

        print("Initialisation complete")

    def import_data(self):
        """
        Imports data for all 25 regions
        :param self:
        :return: postcode_dict: dictionary with the region numbers as key and values corresponding to the postcodes per region
        """

        for i in range(1,15):
            # skip nonexisting region number
            if i == 13:
                continue

            # read in coverage data (distances between postcodes of one region)
            coverage_df = pd.read_excel('Data/coverage{}.xlsx'.format(i), header = None)

            self.coverage_lst.append(coverage_df.to_dict('list'))
            print(self.coverage_lst)

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
            print(self.postcode_dic)
            print(self.pop_dic)

        # number of registered accidents per region
        with open('Data/DataAllRegions.txt'.format(i)) as f:
            lines = f.readlines()


        for i, line in enumerate(lines[1:]):
            temp = re.findall(r'\d+', line)
            values = list(map(int, temp))
            self.accidents.update({values[0]: values[1]})
        print("Accidents: {}".format(self.accidents))

        # location of bases and number of ambulances per base
        with open('Data/xMEXCLP.txt'.format(i)) as f:
            lines = f.readlines()

        for line in lines:
            res = re.split(',  |;|', line)



    def calculate_ttt(self):
        """
        Caluclates the total travel time for an ambulance plus 15 min buffer
        :return: total travel time in ms
        """

