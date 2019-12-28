import argparse

class options:
    def __init__(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("--data_dir", default="./datasets/banknot_authentication.csv")
        self.ap.add_argument("--target",default="CLASS")
        self.ap.add_argument("--test_size", default=0.2, type=float)
        self.ap.add_argument("--dataset_size", default=100, type=int)
        self.ap.add_argument("--p", default=0.7)
        self.ap.add_argument("--msc", default=4)
        self.ap.add_argument("--isOptim", default=False)
    
    def parse(self):
        return vars(self.ap.parse_args())