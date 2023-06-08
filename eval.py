import torch
import torch.optim as optim
from neuralprocess import NeuralProcess

import logging
from tqdm import tqdm

from params import get_params
from utils import set_seed

args = get_params()
file_name = 'test_results-u_{}_{}_{}.log'.format(args.dataset, args.few, args.epoch)
logging.basicConfig(level=logging.INFO, filename=file_name, filemode='a', format='%(asctime)s - %(levelname)s: %(message)s')
logging.info(args)
set_seed(args.seed)
use_cuda = args.gpu >= 0 and torch.cuda.is_available()
model = NeuralProcess(args, use_cuda)
if use_cuda:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()

d = 'cuda:{}'.format(args.gpu)
print(d)

checkpoint = torch.load('./Checkpoints/{}/best_mrr_model_{}.pth'.format(self.args.dataset, self.args.few))
self.model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    model.eval()
    results = model.eval_one_time(eval_type='test')
    
    tqdm.write("Total MRR (filtered): {:.6f}".format(results['total_mrrs']))
    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1s']))
    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3s']))
    tqdm.write("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10s']))

    logging.info("Total MRR (filtered): {:.6f}".format(results['total_mrrs']))
    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1s']))
    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3s']))
    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10s']))
    
    tqdm.write("Total Induc MRR (filtered): {:.6f}".format(results['total_induc_mrrs']))
    tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(1, results['total_induc_hits@1s']))
    tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(3, results['total_induc_hits@3s']))
    tqdm.write("Total Induc Hits (filtered) @ {}: {:.6f}".format(10, results['total_induc_hits@10s']))
    
    logging.info("Total Induc MRR (filtered): {:.6f}".format(results['total_induc_mrrs']))
    logging.info("Total Induc Hits (filtered) @ {}: {:.6f}".format(1, results['total_induc_hits@1s']))
    logging.info("Total Induc Hits (filtered) @ {}: {:.6f}".format(3, results['total_induc_hits@3s']))
    logging.info("Total Induc Hits (filtered) @ {}: {:.6f}".format(10, results['total_induc_hits@10s']))

    tqdm.write("Total Trans MRR (filtered): {:.6f}".format(results['total_trans_mrrs']))
    tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(1, results['total_trans_hits@1s']))
    tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(3, results['total_trans_hits@3s']))
    tqdm.write("Total Trans Hits (filtered) @ {}: {:.6f}".format(10, results['total_trans_hits@10s']))

    logging.info("Total Trans MRR (filtered): {:.6f}".format(results['total_trans_mrrs']))
    logging.info("Total Trans Hits (filtered) @ {}: {:.6f}".format(1, results['total_trans_hits@1s']))
    logging.info("Total Trans Hits (filtered) @ {}: {:.6f}".format(3, results['total_trans_hits@3s']))
    logging.info("Total Trans Hits (filtered) @ {}: {:.6f}".format(10, results['total_trans_hits@10s']))

