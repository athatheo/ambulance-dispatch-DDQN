import Environment

def run():
    env = Environment.environment()
    env.import_data()
    print(env.sample_accidents(14))

run()