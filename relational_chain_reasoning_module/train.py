import os
import time
import logging
from pathlib import Path
from collections import OrderedDict

import torch
import networkx
import numpy as np
from tqdm import tqdm

from relational_chain_reasoning_module.dataloader import DEV_MetaQADataLoader
from answer_filtering_module.dataloader import MetaQADataLoader as AFM_MetaQADataLoader
from relational_chain_reasoning_module.model import Answer_filtering_module, Relational_chain_reasoning_module

if __name__ == "__main__":
    ### Load model Answer filering module
    KG_EMBED_PATH = f'knowledge_graph_embedding_module/kg_embeddings/MetaQA_full/best_checkpoint/%s'
    score_bn_path = KG_EMBED_PATH % 'score_bn.npy'
    head_bn_path = KG_EMBED_PATH % 'head_bn.npy'
    R_path = KG_EMBED_PATH % 'R.npy'
    E_path = KG_EMBED_PATH % 'E.npy'
    entity_dict_path = KG_EMBED_PATH % 'entities_idx.dict'
    relation_dict_path = KG_EMBED_PATH % 'relations_idx.dict'
    batch_size = 128
    qa_traindataset_path = 'QA/MetaQA/qa_train_1hop_copy.txt'
    qa_dataloader = AFM_MetaQADataLoader(entity_embed_path=E_path, entity_dict_path=entity_dict_path,
                                         relation_embed_path=R_path, relation_dict_path=relation_dict_path,
                                         qa_dataset_path=qa_traindataset_path,
                                         batch_size=batch_size)
    entity_embeddings = qa_dataloader.dataset.entity_embeddings
    embedding_dim = entity_embeddings.shape[-1]
    relation_embeddings = qa_dataloader.dataset.relation_embeddings
    relation_dim = relation_embeddings.shape[-1]
    vocab_size = len(qa_dataloader.dataset.word_idx)
    word_dim = 400
    hidden_dim = 200
    fc_hidden_dim = 400
    afm_model = Answer_filtering_module(entity_embeddings=entity_embeddings, embedding_dim=embedding_dim,
                                        vocab_size=vocab_size,
                                        word_dim=word_dim, hidden_dim=hidden_dim, fc_hidden_dim=fc_hidden_dim,
                                        relation_dim=relation_dim,
                                        head_bn_filepath=head_bn_path, score_bn_filepath=score_bn_path)
    AFM_STORE_PATH = f'answer_filtering_module/MetaQA_full_1_hop_Fri_May_12_00_03_15_2023/best_afm_model.pt'
    afm_model = torch.load(AFM_STORE_PATH)
    print("LOADING ANSWER FILTERING MODEL SUCCESSFULLY !!!")
    ### end loading model ==========================

    # ======= Dataloader Preparation ========
    KG_NAME = 'MetaQA'
    assert KG_NAME in ['MetaQA', 'FB15k-237', 'WebquestionsSP-tiny']
    HOPS = 1
    assert HOPS in [1, 2, 3]
    KG_HALF = False
    assert type(KG_HALF) is bool
    if KG_NAME == 'MetaQA':
        QA_PAIRS_PATH = f'QA/MetaQA/{"half_" if KG_HALF else ""}qa_%s_{str(HOPS)}hop.txt'
    else:
        QA_PAIRS_PATH = f'QA/WebQuestionsSP/qa_%s_webqsp.txt'
    qa_traindataset_path = QA_PAIRS_PATH % 'train'
    qa_devdataset_path = QA_PAIRS_PATH % 'dev'
    qa_testdataset_path = QA_PAIRS_PATH % 'test'
    batch_size = 16
    BEST_OR_FINAL = 'best'
    assert BEST_OR_FINAL in ['best', 'final']
    KG_EMBED_PATH = f'knowledge_graph_embedding_module/kg_embeddings/{KG_NAME}{"_half" if KG_HALF else "_full"}/' \
                    f'{BEST_OR_FINAL}_checkpoint/%s'
    E_path = KG_EMBED_PATH % 'E.npy'
    RELATION_EMBEDDINGS_PATH = KG_EMBED_PATH % 'R.npy'
    RELATION_EMBEDDINGS = np.load(RELATION_EMBEDDINGS_PATH)
    ENTITY_DICT_PATH = KG_EMBED_PATH % 'entities_idx.dict'
    RELATION_DICT_PATH = KG_EMBED_PATH % 'relations_idx.dict'
    ENTITY_DICT = dict()
    with open(ENTITY_DICT_PATH, mode='rt', encoding='utf-8') as inp:
        for line in inp:
            split_infos = line.strip().split('\t')
            ENTITY_DICT[split_infos[0]] = split_infos[1]
    RELATION_DICT = dict()
    with open(RELATION_DICT_PATH, mode='rt', encoding='utf-8') as inp:
        for line in inp:
            split_infos = line.strip().split('\t')
            RELATION_DICT[split_infos[0]] = split_infos[1]

    # ========== Result Store ==========
    TRAINING_RESULTS_DIR = os.path.join('relational_chain_reasoning_module',
                                        '_'.join([KG_NAME, "half" if KG_HALF else "full", str(HOPS), 'hop',
                                                  time.asctime().replace(' ', '_').replace(':', '_')]))
    if not os.path.isdir(TRAINING_RESULTS_DIR):
        os.mkdir(TRAINING_RESULTS_DIR)
    best_model_path = os.path.join(TRAINING_RESULTS_DIR, 'best_rcrm_model.pt')
    final_model_path = os.path.join(TRAINING_RESULTS_DIR, 'final_rcrm_model.pt')

    # ========== Logger ==========
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    file_logger = logging.FileHandler(os.path.join(TRAINING_RESULTS_DIR, 'training.log'), encoding='UTF-8')
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_logger)
    logger.info(f'Meta Infos ===> KG name:[{KG_NAME}], Gop:[{HOPS}], Is_half:[{KG_HALF}], '
                f'Relation counts:[{len(RELATION_DICT)}], Entity counts:[{len(ENTITY_DICT)}]')

    # =========== Prepare Knowledge Graph ===========
    KG_GRAPH_PATH = f'knowledge_graph_embedding_module/knowledge_graphs/' \
                    f'{KG_NAME}{"_half" if KG_HALF else "_full"}/%s.txt'

    def load_data(data_path, reverse):
        with open(data_path, 'r', encoding='utf-8') as inp_:
            data = [l.strip().split('\t') for l in inp_.readlines()]
            if reverse:
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
            new_data = []
            for i in data:
                new_data.append([(ENTITY_DICT[i[0]], {'name': i[0]}), {'r_idx': RELATION_DICT[i[1]], 'r': i[1]},
                                 (ENTITY_DICT[i[2]], {'name': i[2]})])
        return new_data

    # add triple to KG
    all_KG_triples = load_data(KG_GRAPH_PATH % 'train', True) + \
                     load_data(KG_GRAPH_PATH % 'valid', True) + \
                     load_data(KG_GRAPH_PATH % 'test', True)
    KG = networkx.DiGraph()
    """
            Use (node, attrdict) tuples to update attributes for specific nodes.
            >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])
            >>> G.nodes[1]["size"]
            11
    """
    nodes_data = []
    for node in list(map(lambda x: (x[0], x[2]), all_KG_triples)):
        nodes_data.extend(node)
    nodes_data = [(int(node[0]), node[1]) for node in nodes_data]
    edges_data = list(map(lambda x: (x[0][0], x[2][0], x[1]), all_KG_triples))
    edges_data = [(int(edge[0]), int(edge[1]), {'r_idx': int(edge[2]['r_idx']), 'r': edge[2]['r']})
                  for edge in edges_data]
    KG.add_nodes_from(nodes_data)  # store all nodes and their 'name' attributes.
    KG.add_edges_from(edges_data)  # store all edges and edges` type [r_idx, r]
    logger.info(f"Number of Edges: {KG.number_of_edges()}, Number of Nodes: {KG.number_of_nodes()}")

    # ========== Load Answer_Filtering_Module ==========
    afm_model.to(device=torch.device('cuda'))
    afm_model.eval()
    word_idx = dict()  # tokenizer
    AFM_STORE_FOLDER = Path(AFM_STORE_PATH).parent.absolute()
    with open(os.path.join(AFM_STORE_FOLDER, 'word_idx.txt'), mode='rt', encoding='utf-8') as inp:
        for line in inp:
            line = line.strip().split('\t')
            word_idx[line[0]] = int(line[1])

    # ========== Load Relational_Chain_Reasoning_Module ==========
    load_from = ''
    rcrm_model = Relational_chain_reasoning_module(relation_dim=RELATION_EMBEDDINGS.shape[-1], dim_l1=768,
                                                   dim_l2=RELATION_EMBEDDINGS.shape[-1] * 2,
                                                   lstm_hidden_dim=RELATION_EMBEDDINGS.shape[-1],
                                                   relation_embeddings=RELATION_EMBEDDINGS)
    if load_from:
        rcrm_model.load_state_dict(torch.load(load_from))
    rcrm_model.to(device=torch.device('cuda'))

    # ============= Prepare Dataset =============
    afm_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH, batch_size=batch_size,
                                          relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_traindataset_path)
    afm_dev_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                              batch_size=batch_size,
                                              relation_dict_path=RELATION_DICT_PATH, qa_dataset_path=qa_devdataset_path)
    afm_test_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=ENTITY_DICT_PATH,
                                               batch_size=batch_size,
                                               relation_dict_path=RELATION_DICT_PATH,
                                               qa_dataset_path=qa_testdataset_path)
    afm_process = tqdm(afm_dataloader, total=len(afm_dataloader), unit=' batches')
    afm_dev_process = tqdm(afm_dev_dataloader, total=len(afm_dev_dataloader), unit=' batches')
    afm_test_process = tqdm(afm_test_dataloader, total=len(afm_test_dataloader), unit=' batches')

    # ============= Hyper-Parameters Preparation =============
    N_EPOCHS = 400
    PATIENCE = 5
    LR = 0.0001
    adam_optimizer = torch.optim.Adam(rcrm_model.parameters(), lr=LR)
    TEST_TOP_K = 1
    LR_DECAY = 0.95
    TEST_INTERVAL = 1
    USE_TOP_K = SUB_BATCH_SIZE = 32
    NO_UPDATE = 0
    BEST_VAL_SCORE = 0

    # ========== Dataset Transformer ==========
    def dataset_trans(batch_ranked_topK_entity, batch_head_entity, batch_answers, text_qs):
        """
        batch_ranked_topK_entity: Obtained batch_size questions in the candidate output entity subscript of the afm model, shape: (batch_size, USE_TOP_K)
        batch_answers: The actual set of correct answers, shape: (batch_size, random=> depends on the number of answers)
        batch_head_entity: The subscript shape of the topic entity corresponding to each question: (batch_size, )
        next step: For the topK candidate entities of each question, retrieve its relational chain relational chain in KG. For topK, those that belong to the answer are marked as positive samples, and those that do not belong to the answer are marked as negative samples
        test_qs : The text of batch_size questions
        send to rcrm_model ===> question_text, relational_chain_idxs, relation_chain_lengths, max_chain_len, label
        """
        rcrm_train_batch = []  # Collect data for rcrm_model training
        for idx, head_entity in enumerate(batch_head_entity.tolist()):
            answers = batch_answers[idx]
            candids = batch_ranked_topK_entity[idx]
            text_q = text_qs[idx]
            for candid in candids:
                # Query the relationship chain between the topK candidate entities <=>topic entities
                try:
                    e_path = networkx.shortest_path(KG, source=head_entity, target=candid)
                except networkx.exception.NetworkXNoPath as ex:
                    # e_path = [head_entity, candid]
                    print(f"NetworkXNoPath: {ex}")
                    continue

                if len(e_path) < 2:
                    continue  # If there is no path for topic and candidate, discard it
                else:
                    relation_chain = []
                    for i in range(len(e_path) - 1):
                        # relation_chain.append((KG.edges[e_path[i], e_path[i+1]]['r_idx'], KG.edges[e_path[i], e_path[i+1]]['r']))
                        relation_chain.append(KG.edges[e_path[i], e_path[i + 1]]['r_idx'])
                if (candid not in answers) or (candid == head_entity):
                    rcrm_train_batch.append([text_q, relation_chain, 0, candid])
                else:  # Only when the candidate is in the answer and not the topic is it marked as a positive sample
                    rcrm_train_batch.append([text_q, relation_chain, 1, candid])
        all_batch_count = len(rcrm_train_batch)
        sorted_batch = list(sorted(rcrm_train_batch, key=lambda x: len(x[1]), reverse=True))
        max_chain_len = len(sorted_batch[0][1])
        final_qs = []
        final_rc = []
        final_rcl = []
        final_label = []
        final_candids = []
        for text_q_, relation_chain_, label, candid_ in sorted_batch:
            final_qs.append(text_q_)
            if len(relation_chain_) > 8:
                final_rc.append(relation_chain_[:8])
                final_rcl.append(8)
            else:
                final_rc.append(relation_chain_ + [0] * (8 - len(relation_chain_)))
                final_rcl.append(len(relation_chain_))
            final_label.append(label)
            final_candids.append(candid_)
        final_rc = torch.tensor(final_rc, device=torch.device('cuda'))
        final_rcl = torch.tensor(final_rcl, device=torch.device('cuda'))
        final_label = torch.tensor(final_label, device=torch.device('cuda'))
        return final_qs, final_rc, final_rcl, max_chain_len, final_label, all_batch_count, final_candids

    # ==== Calculate the hit@1 accuracy of this batch by (question description, similarity, candidate answer)
    # + (question description, real answer) ====
    def calcul_accuracy(batch_similarities, final_qs, final_candids, text_qs, batch_answers):
        predicted_qa = dict()
        for i, j, k in zip(batch_similarities, final_qs, final_candids):
            if j not in predicted_qa:
                predicted_qa[j] = (k, i)
            if i < predicted_qa[j][1]:  # When the similarity is less than the currently stored similarity, replace
                predicted_qa[j] = (k, i)
        correct_count = 0
        wrong_count = 0
        for m, n in zip(text_qs, batch_answers):
            predicted_answer = predicted_qa[m][0]
            if predicted_answer in batch_answers:
                correct_count += 1
            else:
                wrong_count += 1
        accuracy = correct_count / (wrong_count + correct_count)
        return correct_count, wrong_count, accuracy


    # ========== Training Step ===========
    for epoch_idx in range(N_EPOCHS):
        epoch_idx += 1
        avg_epoch_loss = 0
        afm_process.set_description('{}/{}'.format(epoch_idx, N_EPOCHS))

        # For Training Processing
        for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len, text_qs in afm_process:
            batch_ranked_topK_entity = afm_model.get_ranked_top_k(batch_questions_index.cuda(),
                                                                  batch_questions_length.cuda(),
                                                                  batch_head_entity.cuda(), max_sent_len,
                                                                  K=USE_TOP_K).indices.tolist()
            final_qs, final_rc, final_rcl, max_chain_len, final_label, all_batch_count, _ = dataset_trans(
                batch_ranked_topK_entity, batch_head_entity, batch_answers, text_qs)

            loss = rcrm_model(question_text=final_qs, relational_chain_idxs=final_rc, relation_chain_lengths=final_rcl,
                              max_chain_len=max_chain_len, label=final_label, is_test=False)
            loss.backward()
            adam_optimizer.step()
            avg_iter_loss = loss.item() / all_batch_count
            avg_epoch_loss += avg_iter_loss
            afm_process.set_postfix(
                OrderedDict(Epoch=epoch_idx, Batch=all_batch_count, Batch_Loss=loss.item(), avg_Loss=avg_iter_loss))
            afm_process.update()
        logger.info(f'{epoch_idx}-th epoch: average_loss: {avg_epoch_loss / len(afm_dataloader) * batch_size}')

        if epoch_idx % TEST_INTERVAL == 0:
            rcrm_model.eval()
            eval_correct_count = 0
            eval_wrong_count = 0
            # For Dev Processing
            for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len, text_qs in afm_dev_process:
                batch_ranked_topK_entity = afm_model.get_ranked_top_k(batch_questions_index.cuda(),
                                                                      batch_questions_length.cuda(),
                                                                      batch_head_entity.cuda(),
                                                                      max_sent_len, K=USE_TOP_K).indices.tolist()
                final_qs, final_rc, final_rcl, max_chain_len, final_label, all_batch_count, final_candids = dataset_trans(
                    batch_ranked_topK_entity, batch_head_entity, batch_answers, text_qs)
                batch_similarities = rcrm_model(question_text=final_qs, relational_chain_idxs=final_rc,
                                                relation_chain_lengths=final_rcl,
                                                max_chain_len=max_chain_len, is_test=True)
                # batch_similarities. It is the similarity (Euclidean distance) calculation score for all candidates of batch_size.
                batch_similarities = batch_similarities.tolist()
                batch_head_entity = batch_head_entity.tolist()
                correct_count, wrong_count, accuracy = calcul_accuracy(batch_similarities, final_qs,
                                                                       final_candids, text_qs, batch_answers)
                logger.info(f'batch_size:[{batch_size}]. average dev accuracy: [{accuracy}].')
                eval_correct_count += correct_count
                eval_wrong_count += wrong_count

            # For Testing Processing
            for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len, text_qs in afm_test_process:
                batch_ranked_topK_entity = afm_model.get_ranked_top_k(batch_questions_index.cuda(),
                                                                      batch_questions_length.cuda(),
                                                                      batch_head_entity.cuda(),
                                                                      max_sent_len, K=USE_TOP_K).indices.tolist()
                final_qs, final_rc, final_rcl, max_chain_len, final_label, all_batch_count, final_candids = dataset_trans(
                    batch_ranked_topK_entity, batch_head_entity, batch_answers, text_qs)
                batch_similarities = rcrm_model(question_text=final_qs, relational_chain_idxs=final_rc,
                                                relation_chain_lengths=final_rcl,
                                                max_chain_len=max_chain_len, is_test=True)
                # batch_similarities. It is the similarity (Euclidean distance) calculation score for all candidates of batch_size.
                batch_similarities = batch_similarities.tolist()
                batch_head_entity = batch_head_entity.tolist()
                correct_count, wrong_count, accuracy = calcul_accuracy(batch_similarities, final_qs,
                                                                       final_candids, text_qs, batch_answers)
                logger.info(f'batch_size:[{batch_size}]. average test accuracy: [{accuracy}].')
                eval_correct_count += correct_count
                eval_wrong_count += wrong_count

            # Combined accuracy on test and dev
            eval_correct_rate = eval_correct_count / (eval_correct_count + eval_wrong_count)
            if eval_correct_rate > BEST_VAL_SCORE + 0.0001:
                logger.info(
                    f'evaluation accuracy hit@{TEST_TOP_K} increases from {BEST_VAL_SCORE} to {eval_correct_rate}, save the model to {best_model_path}.')
                # torch.save(model.state_dict(), best_model_path)
                torch.save(rcrm_model, best_model_path)
                BEST_VAL_SCORE = eval_correct_rate
                NO_UPDATE = 0
            elif NO_UPDATE >= PATIENCE:
                logger.info(f'Model does not increases in the epoch-[{epoch_idx}], which has exceed patience.')
                exit(-1)
            else:
                NO_UPDATE += 1
            rcrm_model.train()

    logger.info(f"final epoch has reached. stop and save rcrm-model to {final_model_path}.")
    # torch.save(model.state_dict(), final_model_path)
    torch.save(rcrm_model, final_model_path)
    logger.info("bingo.")
