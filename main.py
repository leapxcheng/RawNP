import torch
import torch.optim as optim
from neuralprocess import NeuralProcess

import logging
from tqdm import tqdm

from params import get_params
from utils import set_seed

class Trainer(object):

    def __init__(self, args):

        super(Trainer, self).__init__()

        self.args = args
        # use cuda or not
        self.use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.set_device(args.gpu)
            print('use cuda')

        self.model = NeuralProcess(args, self.use_cuda)
            
        if self.use_cuda:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
                
        self.best_mrr = 0

    def train(self):
        
        self.args.grad_norm = 1.0
        # early-stop strategy
        import time
        patient = 0
        # checkpoint = torch.load('./Checkpoints/{}/6000_best_mrr_model.pth'.format(self.args.dataset+'_100_'+str(self.args.few)))
        # self.model.load_state_dict(checkpoint['state_dict'])
        for epoch in tqdm(range(self.args.n_epochs)):

            self.model.train()
            loss = self.model.forward(epoch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.optimizer.step()

            if epoch % self.args.evaluate_every == 0:
                print("-------------------------------------Valid---------------------------------------")
                with torch.no_grad():
                    start = time.time()
                    self.model.eval()
                    results = self.model.eval_one_time(eval_type='valid')
                    end = time.time()
                    print(end-start)
                    mrr = results['total_mrr']
                    logging.info("Epoch: {} - Validation".format(epoch))
                    logging.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    logging.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
                    print("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
                    print("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
                    if mrr > self.best_mrr:
                        patient = 0
                        self.best_mrr = mrr
                        torch.save({'state_dict': self.model.state_dict(), 'epoch': epoch}, './Checkpoints/{}/best_mrr_model_{}.pth'.format(self.args.dataset, self.args.few))
                    else:
                        patient += 1
                                                                                     
            if patient >= 4:
                break
        
        # For test
        # checkpoint = torch.load('./Checkpoints/{}/best_mrr_model_{}.pth'.format(self.args.dataset, self.args.few))
        # self.model.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            self.model.eval()
            results = self.model.eval_one_time(eval_type='test')
            mrr = results['total_mrr']
            logging.info("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
            logging.info("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))
            print("Total MRR (filtered): {:.6f}".format(results['total_mrr']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(1, results['total_hits@1']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(3, results['total_hits@3']))
            print("Total Hits (filtered) @ {}: {:.6f}".format(10, results['total_hits@10']))


if __name__ == '__main__':

    args = get_params()
    filename = './log/' + args.dataset + '_' + args.score_function + '_np_t'+ str(args.num_train_entity) + '_few' + str(args.few) + '.log'
    logging.basicConfig(level=logging.INFO, filename=filename, filemode='a', format='%(asctime)s - %(levelname)s: %(message)s')
    logging.info(args)
    print(args)
    
    set_seed(args.seed)
    
    trainer = Trainer(args)
    trainer.train()