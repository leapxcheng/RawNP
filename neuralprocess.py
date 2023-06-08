import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

import numpy as np
import random
import os
import json
import copy
import logging
import utils
from tqdm import tqdm
# import dgl

from models import Decoder, LatentEncoder, MuSigmaEncoder, EntityEmbedding
from utils import load_data, load_processed_data

class NeuralProcess(nn.Module):
   
    def __init__(self, args, use_cuda):
    
        super(NeuralProcess, self).__init__()

        self.args = args
        self.use_cuda = use_cuda

        self.entity2id, self.relation2id, self.train_triplets, self.valid_triplets, self.test_triplets = load_data('./Dataset/raw_data/{}'.format(args.dataset))

        self.meta_train_task_entity_to_triplets, self.meta_valid_task_entity_to_triplets, self.meta_test_task_entity_to_triplets \
            = load_processed_data('./Dataset/processed_data/{}'.format(self.args.dataset))

        self.all_triplets = torch.LongTensor(np.concatenate((self.train_triplets, self.valid_triplets, self.test_triplets)))
        
        
        self.meta_task_entity = np.concatenate((list(self.meta_train_task_entity_to_triplets.keys()),
                                    list(self.meta_valid_task_entity_to_triplets.keys()),
                                    list(self.meta_test_task_entity_to_triplets.keys())))
        # not hold an entity during training. 
        self.entities_list = np.delete(np.arange(len(self.entity2id)), self.meta_task_entity)

        self.load_pretrain_embedding()
        self.embed = EntityEmbedding(self.args.embed_size, self.args.embed_size, len(self.entity2id), len(self.relation2id),
                            args = self.args, entity_embedding = self.pretrain_entity_embedding, relation_embedding = self.pretrain_relation_embedding)
        # MLP
        self.latent_encoder = LatentEncoder(embed_size=args.embed_size, num_hidden1=500, num_hidden2=200,
                                r_dim=args.embed_size, dropout_p=0.5, rw=self.args.rw)
        # MuSigma
        self.dist = MuSigmaEncoder(args.embed_size, args.embed_size)

        # Decoder
        self.decoder = Decoder(self.args, self.args.embed_size)
        
        # Construct Graph
        # self.g, self.rel = self.construct_graph()
        # Anonymous Random Walk Encoder
        self.arw_encoder = nn.RNN(1, self.args.rw, 1, batch_first=True)
        file = '{}_rw_{}.json'.format(self.args.dataset, self.args.nums_rw)
        with open(file) as f:
            self.random_walks = json.load(f)
        self.anonymous_walk = torch.FloatTensor([node[1] for node in self.random_walks])
        # We set the embedding of unseen entities as the zero vector.
        meta_task_entity = torch.LongTensor(self.meta_task_entity)
        self.embed.entity_embedding.weight.data[meta_task_entity] = torch.zeros(len(meta_task_entity), self.embedding_size)

        self.device = 'cuda:{}'.format(self.args.gpu)
        if self.use_cuda:
            self.all_triplets = self.all_triplets.to(self.device)

    def construct_graph(self):
        # 所有三元组
        all_triplets = np.array(self.all_triplets)

        # 从原本的KG图中删除验证集和测试集中含有的节点，得到KG'
        train_entity = list(set(self.meta_train_task_entity_to_triplets.keys()))
        valid_entity = list(set(self.meta_valid_task_entity_to_triplets.keys()))
        test_entity = list(set(self.meta_test_task_entity_to_triplets.keys()))
        delete_entity = valid_entity + test_entity

        mask = None
        for entity in delete_entity:
            head_mask = all_triplets[:, 0] == entity
            tail_mask = all_triplets[:, 2] == entity
            if mask is None:
                mask = head_mask | tail_mask
            else:
                mask = mask | head_mask | tail_mask
        all_triplets = all_triplets[~mask]

        src, rel, dst = all_triplets.transpose()
        uniq_v, _ = np.unique((src, dst), return_inverse=True)
        rel = np.append(rel, [-1])

        g = dgl.graph((src, dst))

        return g, rel

    def load_pretrain_embedding(self):

        # Set the embedding dimension for both entity and relation as 100
        self.embedding_size = int(self.args.pre_train_emb_size)
        if self.args.pre_train:

            pretrain_model_path = './Pretraining/{}'.format(self.args.dataset)

            entity_file_name = os.path.join(pretrain_model_path, '{}_entity_{}.npy'.format(self.args.pre_train_model, self.embedding_size))
            relation_file_name = os.path.join(pretrain_model_path, '{}_relation_{}.npy'.format(self.args.pre_train_model, self.embedding_size))

            # pretrain_entity_embedding: [num_entities, 100]
            # pretrain_relation_embedding: [num_relations, 100]
            self.pretrain_entity_embedding = torch.Tensor(np.load(entity_file_name))
            self.pretrain_relation_embedding = torch.Tensor(np.load(relation_file_name))
        else:
            self.pretrain_entity_embedding = None
            self.pretrain_relation_embedding = None

    # # Meta-Learning for Long-Tail Tasks
    def cal_train_few(self, epoch):
        if self.args.model_tail == 'log':
            for i in range(self.args.max_few):
                if epoch < (self.args.n_epochs / (2 ** i)):
                    continue
                return max(min(self.args.max_few, self.args.few + i - 1), self.args.few)
            return self.args.max_few
        else:
            return self.args.few

    def concat_one_unseen_embed(self, unseen, unseen_embedding, rw_embed, triplets, flag):
        
        h, r, t = torch.t(triplets)

        if self.use_cuda:
            h = h.cuda()
            t = t.cuda()
        
        h_embed = self.embed.entity_embedding(h)
        t_embed = self.embed.entity_embedding(t)
        r_embed = self.embed.relation_embedding[r]

        h = h.cpu().numpy()
        t = t.cpu().numpy()
        h_idx = np.where(h == unseen)[0]
        t_idx = np.where(t == unseen)[0]
        if h_idx.shape[0] != 0:
            h_embed[h_idx] = unseen_embedding
        if t_idx.shape[0] != 0:
            t_embed[t_idx] = unseen_embedding
        
        if flag == 1:       # positive triplets
            label = torch.ones(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([rw_embed, h_embed, r_embed, t_embed, label], dim=-1)
        elif flag == 0:     # negative triplets
            label = torch.zeros(triplets.shape[0], 1).to(h_embed)
            embeddings = torch.cat([rw_embed, h_embed, r_embed, t_embed, label], dim=-1)
        else:
            embeddings = torch.cat([rw_embed, h_embed, r_embed, t_embed], dim=-1)
        
        return embeddings

    def score_loss(self, unseen_entity, unseen_entity_embedding, triplets, target, use_cuda, z=None, rw=None):

        head_embeddings = self.embed.entity_embedding(triplets[:, 0])
        relation_embeddings = self.embed.relation_embedding[triplets[:, 1]]
        tail_embeddings = self.embed.entity_embedding(triplets[:, 2])

        head_embeddings[triplets[:, 0] == unseen_entity] = unseen_entity_embedding
        tail_embeddings[triplets[:, 2] == unseen_entity] = unseen_entity_embedding

        len_positive_triplets = int(len(target) / (self.args.negative_sample + 1))

        # score = self.decoder(head_embeddings, relation_embeddings, tail_embeddings, z)
        # positive_score = score[:len_positive_triplets]
        # negative_score = score[len_positive_triplets:]

        # add z
        head_embeddings = self.decoder(head_embeddings, z, rw)
        tail_embeddings = self.decoder(tail_embeddings, z, rw)
        # head_embeddings, tail_embeddings = self.decoder(head_embeddings, tail_embeddings, z)

        if self.args.score_function == 'DistMult':

            score = head_embeddings * relation_embeddings * tail_embeddings
            score = torch.sum(score, dim = 1)

            positive_score = score[:len_positive_triplets]
            negative_score = score[len_positive_triplets:]

        elif self.args.score_function == 'TransE':

            pos_head_embeddings = head_embeddings[:len_positive_triplets]
            pos_relation_embeddings = relation_embeddings[:len_positive_triplets]
            pos_tail_embeddings = tail_embeddings[:len_positive_triplets]

            x = pos_head_embeddings + pos_relation_embeddings - pos_tail_embeddings
            positive_score = - torch.norm(x, p = 2, dim = 1)

            neg_head_embeddings = head_embeddings[len_positive_triplets:]
            neg_relation_embeddings = relation_embeddings[len_positive_triplets:]
            neg_tail_embeddings = tail_embeddings[len_positive_triplets:]

            x = neg_head_embeddings + neg_relation_embeddings - neg_tail_embeddings
            negative_score = - torch.norm(x, p = 2, dim = 1)

        else:

            raise ValueError("Score Function Name <{}> is Wrong".format(self.score_function))

        y = torch.ones(len_positive_triplets * self.args.negative_sample)
        if use_cuda:
            y = y.cuda()

        positive_score = positive_score.repeat(self.args.negative_sample)
        # positive_score = positive_score.view(-1, 1).repeat(1, self.args.negative_sample).view(-1)
        loss = F.margin_ranking_loss(positive_score, negative_score, y, margin = self.args.margin)

        return loss

    def randomwalk(self, triplets, unseen):
        keys_pool = self.random_walks.keys()
        total_rw = []
        for idx in range(len(triplets)):
            node = triplets[idx][0] if triplets[idx][2] == unseen else triplets[idx][2]
            if str(node) in keys_pool:
                sample = np.random.choice(10, 5)
                arw_paths = np.array(self.random_walks[str(node)])[sample]

                input = torch.FloatTensor(arw_paths).transpose(1, 0).unsqueeze(-1).to(self.device)
                output, _ = self.arw_encoder(input)
                total_rw.append(output.mean(0).mean(0))
            else:
                total_rw.append(torch.zeros(self.args.rw).to(self.device))

        total_rw = torch.cat(total_rw).view(-1, self.args.rw)

        return total_rw
    
    def random_walk1(self, triplets, unseen):
        total_rw = []
        for idx in range(len(triplets)):
            node = triplets[idx][0] if triplets[idx][2] == unseen else triplets[idx][2]
            sample = np.random.choice(10, 5)
            total_rw.append(self.anonymous_walk[node][sample])

        total_rw = torch.cat(total_rw, dim=0).unsqueeze(-1)

        return total_rw

    def random_walk2(self, entities):
        total_rw = []

        walks = self.anonymous_walk[entities]
        for node in walks:
            sample = np.random.choice(10, 5)
            total_rw.append(node[sample])

        total_rw = torch.cat(total_rw, dim=0).unsqueeze(-1)
        return total_rw

    def forward(self, epoch):
        train_task_pool = list(self.meta_train_task_entity_to_triplets.keys()) # filter < K + q
        random.shuffle(train_task_pool)       

        total_unseen_entity = []
        total_unseen_entity_embedding = []

        support_pos_triplets = []
        support_neg_triplets = []
        query_pos_triplets = []
        query_neg_triplets = []

        context_dists = []
        target_dists = []

        query_counts = []

        total_loss = 0
        total_kl_loss = 0
        train_few = self.cal_train_few(epoch)
        # train_few = 5
        for unseen_entity in train_task_pool[:self.args.num_train_entity]:
            
            # randomly sample a unseen entities and its corresponding triplets.
            triplets = self.meta_train_task_entity_to_triplets[unseen_entity]
            random.shuffle(triplets)

            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()
            
            train_triplets = triplets[:train_few]   # 10 default
            test_triplets = triplets[train_few:]
            if (len(triplets)) - train_few < 1:
                    continue

            entities_list = self.entities_list  # including entites except the meta-testing.
            false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))
            
            # # Support set
            s_false_entities = np.random.choice(false_candidates, size=train_few * self.args.negative_sample)
            s_neg_samples = np.tile(train_triplets, (self.args.negative_sample, 1))
            s_neg_samples[s_neg_samples[:, 0] == unseen_entity, 2] = s_false_entities[s_neg_samples[:, 0] == unseen_entity]
            s_neg_samples[s_neg_samples[:, 2] == unseen_entity, 0] = s_false_entities[s_neg_samples[:, 2] == unseen_entity]  
            support_pos_triplets.extend(train_triplets)
            support_neg_triplets.extend(s_neg_samples)

            ## The part of baseline
            # Query set
            q_false_entities = np.random.choice(false_candidates, size=(len(triplets) - train_few) * self.args.negative_sample)
            q_neg_samples = np.tile(test_triplets, (self.args.negative_sample, 1))
            q_neg_samples[q_neg_samples[:, 0] == unseen_entity, 2] = q_false_entities[q_neg_samples[:, 0] == unseen_entity]
            q_neg_samples[q_neg_samples[:, 2] == unseen_entity, 0] = q_false_entities[q_neg_samples[:, 2] == unseen_entity]  
            query_pos_triplets.extend(test_triplets)
            query_neg_triplets.extend(q_neg_samples)
            query_counts.append(len(test_triplets))

            # 计算unseen entity的embeddings
            unseen_entity_embedding = self.embed(unseen_entity, train_triplets, self.use_cuda)
            total_unseen_entity.append(unseen_entity)
            total_unseen_entity_embedding.append(unseen_entity_embedding)

            # Support set
            support_pos_triplet = torch.LongTensor(train_triplets)
            support_neg_triplet = torch.LongTensor(s_neg_samples)
            query_pos_triplet = torch.LongTensor(test_triplets)
            query_neg_triplet = torch.LongTensor(q_neg_samples)

            rw_total_triplets = np.concatenate((train_triplets, s_neg_samples, test_triplets, q_neg_samples))
            total_rw = self.random_walk1(rw_total_triplets, unseen_entity)
            output, _ = self.arw_encoder(total_rw.cuda())
            
            q1, q2, q3 = train_triplets.shape[0]*5, s_neg_samples.shape[0]*5, test_triplets.shape[0]*5
            support_pos_rw = output[:q1].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            support_neg_rw = output[q1:q1+q2].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            query_pos_rw = output[q1+q2:q1+q2+q3].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            query_neg_rw = output[q1+q2+q3:].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            query_rw = torch.cat([query_pos_rw, query_neg_rw], dim=0)

            # 仅修改当前triplets中含有的该unsseen节点的embeddings
            s_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_pos_rw, support_pos_triplet, 1)
            s_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_neg_rw, support_neg_triplet, 0)
            q_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_pos_rw, query_pos_triplet, 1)
            q_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, query_neg_rw, query_neg_triplet, 0)

            # Encoder
            s_pos_r = self.latent_encoder(s_emb_p)
            s_neg_r = self.latent_encoder(s_emb_n)
            q_pos_r = self.latent_encoder(q_emb_p)
            q_neg_r = self.latent_encoder(q_emb_n)
            # for each unseen entity.
            c_r = torch.cat([s_pos_r, s_neg_r], dim=0)    # prior
            t_r = torch.cat([s_pos_r, s_neg_r, q_pos_r, q_neg_r], dim=0)   # posteior
            c_r = torch.mean(c_r, dim=0, keepdim=True)
            t_r = torch.mean(t_r, dim=0, keepdim=True)

            context_dist = self.dist(c_r)
            target_dist = self.dist(t_r)
            z = target_dist.rsample()

            # kld_loss = kl(posterior, prior)
            kld = kl_divergence(target_dist, context_dist)
            kld = kld.sum()
            if total_kl_loss == 0:
                total_kl_loss = kld
            else:
                total_kl_loss += kld

            samples = np.concatenate([test_triplets, q_neg_samples])
            labels = np.zeros(len(samples), dtype=np.float32)
            labels[:len(test_triplets)] = 1
            samples = torch.LongTensor(samples)
            labels = torch.LongTensor(labels)
            if self.use_cuda:
                samples = samples.cuda()
                labels = labels.cuda()

            losss = self.score_loss(unseen_entity, unseen_entity_embedding, samples, target=labels, use_cuda=self.use_cuda, z=z, rw=query_rw)
            
            if total_loss == 0:
                total_loss = losss
            else:
                total_loss += losss

        ranking_loss = total_loss / self.args.num_train_entity
        kl_loss = total_kl_loss / self.args.num_train_entity
        loss = ranking_loss + kl_loss
        
        if (epoch + 1) % 50 == 0:
            logging.info("Epoch: {} \t loss: {:.4f} \t ranking_loss: {:.4f} \t kl_loss: {:.4f}".format(epoch, loss, ranking_loss, kl_loss))
            print("Epoch: {} \t loss: {:.4f} \t ranking_loss: {:.4f} \t kl_loss: {:.4f}".format(epoch, loss, ranking_loss, kl_loss))

        return loss
    
    def eval_one_time(self, eval_type):
        
        if eval_type == 'valid':
            test_task_dict = self.meta_valid_task_entity_to_triplets
            test_task_pool = list(self.meta_valid_task_entity_to_triplets.keys())
        elif eval_type == 'test':
            test_task_dict = self.meta_test_task_entity_to_triplets
            test_task_pool = list(self.meta_test_task_entity_to_triplets.keys())
        else:
            raise ValueError("Eval Type <{}> is Wrong".format(eval_type))

        total_ranks = []
        total_subject_ranks = []
        total_object_ranks = []
        total_unseen_ranks = []

        total_unseen_entity = []
        total_unseen_entity_embedding = []
        
        total_test_triplets_dict = {}

        support_pos_triplets = []
        support_neg_triplets = []
        query_triplets = []

        context_dists = []
        target_dists = []

        total_ranks = []
        total_induc_ranks = []
        total_trans_ranks = []

        query_counts = []
        cnt = 0

        for unseen_entity in tqdm(test_task_pool):
    
            triplets = test_task_dict[unseen_entity]
            triplets = np.array(triplets)
            heads, relations, tails = triplets.transpose()

            train_triplets = triplets[:self.args.few]
            test_triplets = triplets[self.args.few:]

            entities_list = self.entities_list  # including entites except the meta-testing.
            false_candidates = np.array(list(set(entities_list) - set(np.concatenate((heads, tails)))))

            if (len(triplets)) - self.args.few < 1:
                continue
            # Support set
            s_false_entities = np.random.choice(false_candidates, size=self.args.few * self.args.negative_sample)
            s_neg_samples = np.tile(train_triplets, (self.args.negative_sample, 1))
            s_neg_samples[s_neg_samples[:, 0] == unseen_entity, 2] = s_false_entities[s_neg_samples[:, 0] == unseen_entity]
            s_neg_samples[s_neg_samples[:, 2] == unseen_entity, 0] = s_false_entities[s_neg_samples[:, 2] == unseen_entity]  
            
            # query_triplets.extend(test_triplets)
            test_triplets = torch.LongTensor(test_triplets)
            if self.use_cuda:
                test_triplets = test_triplets.to(self.device)

            # Train (Inductive)
            unseen_entity_embedding = self.embed(unseen_entity, train_triplets, use_cuda = self.use_cuda)
            total_unseen_entity.append(unseen_entity)

            support_pos_triplet = torch.LongTensor(train_triplets)
            support_neg_triplet = torch.LongTensor(s_neg_samples)
            # anonymous random walk
            # support_pos_rw = self.randomwalk(train_triplets, unseen_entity)
            # support_neg_rw = self.randomwalk(s_neg_samples, unseen_entity)
            rw_total_triplets = np.concatenate((train_triplets, s_neg_samples))
            total_rw = self.random_walk1(rw_total_triplets, unseen_entity)
            output, _ = self.arw_encoder(total_rw.to(self.device))

            q1 = train_triplets.shape[0]*5
            support_pos_rw = output[:q1].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            support_neg_rw = output[q1:].view(-1, 5, 10, self.args.rw).mean(1).mean(1)
            
            s_emb_p = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_pos_rw, support_pos_triplet, 1)
            s_emb_n = self.concat_one_unseen_embed(unseen_entity, unseen_entity_embedding, support_neg_rw, support_neg_triplet, 0)

            # Encoder
            s_pos_r = self.latent_encoder(s_emb_p)
            s_neg_r = self.latent_encoder(s_emb_n)
            # for each unseen entity.
            c_r = torch.cat([s_pos_r, s_neg_r], dim=0)
            c_r = torch.mean(c_r, dim=0)
            context_dist = self.dist(c_r)
            z = context_dist.rsample()
            
            ranks, ranks_1, ranks_2 = utils.calc_induc_mrr(test_task_pool, self.args, self.random_walk2, self.arw_encoder, self.decoder, z, unseen_entity, unseen_entity_embedding, self.embed.entity_embedding.weight, self.embed.relation_embedding, test_triplets, self.all_triplets, self.use_cuda, score_function=self.args.score_function)
            
            total_ranks.extend(ranks)
            total_induc_ranks.extend(ranks_1)
            total_trans_ranks.extend(ranks_2)

        total_ranks = torch.LongTensor(total_ranks).view(-1)
        total_induc_ranks = torch.LongTensor(total_induc_ranks).view(-1)
        total_trans_ranks = torch.LongTensor(total_trans_ranks).view(-1)


        results = {}

        results['total_mrrs'] = torch.mean(1.0 / total_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_ranks <= hit).float())
            results['total_hits@{}s'.format(hit)] = avg_count.item()

        results['total_induc_mrrs'] = torch.mean(1.0 / total_induc_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_induc_ranks <= hit).float())
            results['total_induc_hits@{}s'.format(hit)] = avg_count.item()

        results['total_trans_mrrs'] = torch.mean(1.0 / total_trans_ranks.float()).item()

        for hit in [1, 3, 10]:
            avg_count = torch.mean((total_trans_ranks <= hit).float())
            results['total_trans_hits@{}s'.format(hit)] = avg_count.item()

        return results