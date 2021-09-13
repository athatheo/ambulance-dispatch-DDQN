import pandas as pd
import openpyxl

# read directly into dictionary?
coverage_df = pd.read_excel('Data/coverage14.xlsx', header = None)
pop_df = pd.read_excel('Data/population14.xlsx', header = None)

dic = coverage_df.to_dict('list')

# merge postcode with distances
# calculate total population per region
