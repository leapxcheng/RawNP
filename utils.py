import random
import torch
import os
import tqdm
import pickle
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Data load and Preprocess
def load_data(file_path):
    '''
        argument:
            file_path: ./Dataset/raw_data/FB15k-237
        
        return:
            entity2id, relation2id, train_triplets, valid_triplets, test_triplets
    '''

    print("load data from {}".format(file_path))

    with open(os.path.join(file_path, 'entity2id.txt')) as f:
        entity2id = dict()

        for line in f:
            entity, eid = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(file_path, 'relation2id.txt')) as f:
        relation2id = dict()

        for line in f:
            relation, rid = line.strip().split('\t')
            relation2id[relation] = int(rid)

    train_triplets = read_triplets(os.path.join(file_path, 'train.txt'), entity2id, relation2id)
    valid_triplets = read_triplets(os.path.join(file_path, 'valid.txt'), entity2id, relation2id)
    test_triplets = read_triplets(os.path.join(file_path, 'test.txt'), entity2id, relation2id)

    print('num_entity: {}'.format(len(entity2id)))
    print('num_relation: {}'.format(len(relation2id)))
    print('num_train_triples: {}'.format(len(train_triplets)))
    print('num_valid_triples: {}'.format(len(valid_triplets)))
    print('num_test_triples: {}'.format(len(test_triplets)))

    return entity2id, relation2id, train_triplets, valid_triplets, test_triplets

def read_triplets(file_path, entity2id, relation2id):
    
    triplets = []

    with open(file_path) as f:
        for line in f:
            head, relation, tail = line.strip().split('\t')
            triplets.append((entity2id[head], relation2id[relation], entity2id[tail]))

    return np.array(triplets)

def load_processed_data(file_path):

    with open(os.path.join(file_path, 'filtered_triplets.pickle'), 'rb') as f:
        filtered_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_train_task_triplets.pickle'), 'rb') as f:
        meta_train_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_triplets.pickle'), 'rb') as f:
        meta_valid_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_triplets.pickle'), 'rb') as f:
        meta_test_task_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_train_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_train_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_valid_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_valid_task_entity_to_triplets = pickle.load(f)

    with open(os.path.join(file_path, 'meta_test_task_entity_to_triplets.pickle'), 'rb') as f:
        meta_test_task_entity_to_triplets = pickle.load(f)

    # return filtered_triplets, meta_train_task_triplets, meta_valid_task_triplets, meta_test_task_triplets, \
    #         meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets

    return meta_train_task_entity_to_triplets, meta_valid_task_entity_to_triplets, meta_test_task_entity_to_triplets


def sort_and_rank(score, target):
    x, indices = torch.sort(score, dim=1, descending=True)  # 按行降序排列
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def calc_induc_mrr(pool, args, rw_func, rw_encoder, decoder, z, unseen_entity, unseen_entity_embedding, all_entity_embeddings, all_relation_embeddings, test_triplets, all_triplets, use_cuda, score_function='DistMult'):
    
    num_entity = len(all_entity_embeddings)
    subject_count = 0
    object_count = 0
        
    ranks = []
    ranks_s = []
    ranks_o = []
    ranks_un = []

    total_ranks = []
    subject_ranks = []
    object_ranks = []
    total_induc_ranks = []
    subject_induc_ranks = []
    object_induc_ranks = []
    total_trans_ranks = []
    subject_trans_ranks = []
    object_trans_ranks = []

    head_relation_triplets = all_triplets[:, :2]
    tail_relation_triplets = torch.stack((all_triplets[:, 2], all_triplets[:, 1])).transpose(0, 1)
    
    d = 'cuda:{}'.format(args.gpu)
    device = torch.device(d)
    
    for test_triplet in test_triplets:
        # import time
        # s = time.time()
        is_trans = False
        is_subject = False
        is_object = False
        
        if (test_triplet[0] in pool) and (test_triplet[2] in pool):
            is_trans = True

        if test_triplet[0] == unseen_entity:

            subject_count += 1
            is_subject = True

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            subject_relation = torch.LongTensor([subject, relation])
            if use_cuda:
                subject_relation = subject_relation.to(device)

            delete_index = torch.sum(head_relation_triplets == subject_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()
            
            if use_cuda:
                # device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 2].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 2].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((object_.view(-1), perturb_entity_index))

            # rnn - candidate
            arw = rw_func(perturb_entity_index)
            arw_emb, _ = rw_encoder(arw.to(device))
            # arw_emb, _ = rw_encoder(arw.to(device))
            arw_emb = arw_emb.view(-1, 5, 10, args.rw).mean(1).mean(1)

            # Score
            if score_function == 'DistMult':
                
                emb_ar = decoder(unseen_entity_embedding, z, arw_emb) * all_relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = decoder(all_entity_embeddings[perturb_entity_index], z, arw_emb)
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                
            elif score_function == 'TransE':
                
                # one decoder h/t = h/t + mlp(z)
                head_embedding = decoder(unseen_entity_embedding, z, arw_emb)
                relation_embedding = all_relation_embeddings[relation]
                tail_embeddings = decoder(all_entity_embeddings[perturb_entity_index], z, arw_emb)

                score = - torch.norm((head_embedding + relation_embedding - tail_embeddings), p = 2, dim = 1)
                score = score.view(1, -1)

            else:

                raise TypeError

        elif test_triplet[2] == unseen_entity:

            object_count += 1
            is_object = True

            subject = test_triplet[0]
            relation = test_triplet[1]
            object_ = test_triplet[2]

            object_relation = torch.LongTensor([object_, relation])
            if use_cuda:
                object_relation = object_relation.to(device)

            delete_index = torch.sum(tail_relation_triplets == object_relation, dim = 1)
            delete_index = torch.nonzero(delete_index == 2).squeeze()

            if use_cuda:
                # device = torch.device('cuda')
                delete_entity_index = all_triplets[delete_index, 0].view(-1).cpu().numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index).to(device)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))
            else:
                delete_entity_index = all_triplets[delete_index, 0].view(-1).numpy()
                perturb_entity_index = np.array(list(set(np.arange(num_entity)) - set(delete_entity_index)))
                perturb_entity_index = torch.from_numpy(perturb_entity_index)
                perturb_entity_index = torch.cat((subject.view(-1), perturb_entity_index))
            
            # rnn - candidate
            arw = rw_func(perturb_entity_index)
            arw_emb, _ = rw_encoder(arw.to(device))
            # arw_emb, _ = rw_encoder(arw.to(device))
            arw_emb = arw_emb.view(-1, 5, 10, args.rw).mean(1).mean(1)

            if score_function == 'DistMult':

                emb_ar = decoder(unseen_entity_embedding, z, arw_emb) * all_relation_embeddings[relation]
                emb_ar = emb_ar.view(-1, 1, 1)

                emb_c = decoder(all_entity_embeddings[perturb_entity_index], z)
                emb_c = emb_c.transpose(0, 1).unsqueeze(1)

                out_prod = torch.bmm(emb_ar, emb_c)
                score = torch.sum(out_prod, dim = 0)
                
            elif score_function == 'TransE':

                # one decoder h/t = h/t + mlp(z)
                head_embeddings = decoder(all_entity_embeddings[perturb_entity_index], z, arw_emb)
                relation_embedding = all_relation_embeddings[relation]
                tail_embedding = decoder(unseen_entity_embedding, z, arw_emb)
                
                score = head_embeddings + relation_embedding - tail_embedding
                score = - torch.norm(score, p = 2, dim = 1)
                score = score.view(1, -1)

            else:

                raise TypeError

        else:
            
            raise TypeError
        
         # Cal Rank
        if use_cuda:
            target = torch.tensor(0).to(device)
            rank = sort_and_rank(score, target)
        else:
            target = torch.tensor(0)
            rank = sort_and_rank(score, target)

        rank += 1
        total_ranks.append(rank)
        if is_subject:
            subject_ranks.append(rank)
        elif is_object:
            object_ranks.append(rank)
        
        if is_trans:
            total_trans_ranks.append(rank)
            if is_subject:
                subject_trans_ranks.append(rank)
            elif is_object:
                object_trans_ranks.append(rank)
        else:
            total_induc_ranks.append(rank)
            if is_subject:
                subject_induc_ranks.append(rank)
            elif is_object:
                object_induc_ranks.append(rank)

    return total_ranks, total_induc_ranks, total_trans_ranks
