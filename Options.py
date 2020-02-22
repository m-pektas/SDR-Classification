import argparse


class options:
    def __init__(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument(
            "--data_dir", default="./datasets/banknot_authentication.csv")
        self.ap.add_argument(
            "--target", help="Attribute that will predict", default="CLASS")
        self.ap.add_argument(
            "--test_size", help="Test dataset rate", default=0.2, type=float)
        self.ap.add_argument("--dataset_size", default=100, type=int)
        self.ap.add_argument(
            "--p", help="Majority class threshold.", default=0.7)
        self.ap.add_argument(
            "--msc", help="Minimum sample size threshold.", default=4)
        self.ap.add_argument(
            "--isOptim", help="Do you want to optimize P and MSC parameters ?", default=False)

    def parse(self):
        self.args = vars(self.ap.parse_args())
        self.show_args()
        return self.args

    def show_args(self):
        print("\n\n[INFO] - Parameters\n")
        for k, v in self.args.items():
            print(k, ":", v)
