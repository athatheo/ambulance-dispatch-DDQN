import shelve
import matplotlib.pyplot as plt
import pandas as pd


class Visualiser(object):
    def __init__(self):
        model_data = shelve.open('model.txt')
        self.rewards = model_data['rewards']
        self.model = model_data['model']
        model_data.close()

    def plot_rolling_average(self, windows, region_nr):
        if len(self.rewards[region_nr][1:]) < 1:
            pass
        else:
            plt.plot(pd.DataFrame(self.rewards[region_nr][1:]).rolling(windows).mean())
            plt.title("Region: " + str(region_nr))
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.show()

    def scatter(self, region_nr):
        plt.scatter(range(len(self.rewards[region_nr])), self.rewards[region_nr])
        plt.title("Region: " + str(region_nr))
        plt.show()

    def show_plots_for_all_regions(self):
        for region_nr in range(24):
            if region_nr == 0 or region_nr == 13 or region_nr == 14:
                continue
            self.plot_rolling_average(500, region_nr)
