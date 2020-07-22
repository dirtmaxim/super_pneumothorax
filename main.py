from train_utils import SegTrainer
from matplotlib.pylab import rcParams
import argparse

rcParams['figure.figsize'] = 7, 7

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
args.distributed = False
trainer = SegTrainer('./configs/fold1.yaml', args=args)
trainer.fit()
