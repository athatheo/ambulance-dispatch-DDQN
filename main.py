import Environment

def run():
    env = Environment.Environment()
    env.import_data()
    sum = 0
    for i in range(10000):
        if env.sample_accidents(12).count(1)>0:
            sum = sum + 1

    print(sum)


run()