import os
import time
import logging
from copy import deepcopy
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm

from answer_filtering_module.model import Answer_filtering_module
from answer_filtering_module.dataloader import MetaQADataLoader, DEV_MetaQADataLoader

if __name__ == "__main__":
    if not torch.cuda.is_available:
        print('Sorry, you should buy an NVIDIA Graphic Processing Unit and set CUDA environment!')
        exit(-1)
    else:
        print(f"GPU CHECKING: {torch.cuda.get_device_name(0)}")

    # ============================
    KG_NAME = 'MetaQA'
    assert KG_NAME in ['MetaQA', 'FB15k-237', 'WebquestionsSP-tiny']

    # ============================
    HOPS = 1
    assert HOPS in [1, 2, 3]

    # ============================
    KG_HALF = False
    assert type(KG_HALF) is bool

    # ============================
    if KG_NAME == 'MetaQA':
        QA_PAIRS_PATH = f'QA/MetaQA/{"half_" if KG_HALF else ""}qa_%s_{str(HOPS)}hop_copy.txt'
    else:
        QA_PAIRS_PATH = f'QA/WebQuestionsSP/qa_%s_webqsp.txt'
    qa_traindataset_path = QA_PAIRS_PATH % 'train'
    qa_devdataset_path = QA_PAIRS_PATH % 'dev'
    qa_testdataset_path = QA_PAIRS_PATH % 'test'

    # ======== Dataloader Preparation =========
    BEST_OR_FINAL = 'best'
    assert BEST_OR_FINAL in ['best', 'final']
    KG_EMBED_PATH = f'knowledge_graph_embedding_module/kg_embeddings/{KG_NAME}{"_half" if KG_HALF else "_full"}/' \
                    f'{BEST_OR_FINAL}_checkpoint/%s'
    score_bn_path = KG_EMBED_PATH % 'score_bn.npy'
    head_bn_path = KG_EMBED_PATH % 'head_bn.npy'
    R_path = KG_EMBED_PATH % 'R.npy'
    E_path = KG_EMBED_PATH % 'E.npy'
    entity_dict_path = KG_EMBED_PATH % 'entities_idx.dict'
    relation_dict_path = KG_EMBED_PATH % 'relations_idx.dict'
    batch_size = 256

    qa_dataloader = MetaQADataLoader(entity_embed_path=E_path, entity_dict_path=entity_dict_path, relation_embed_path=R_path
                                     , relation_dict_path=relation_dict_path, qa_dataset_path=qa_traindataset_path,
                                     batch_size=batch_size)
    word_idx = qa_dataloader.dataset.word_idx
    qa_dev_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=entity_dict_path,
                                             relation_dict_path=relation_dict_path,
                                             qa_dataset_path=qa_devdataset_path)
    qa_test_dataloader = DEV_MetaQADataLoader(word_idx=word_idx, entity_dict_path=entity_dict_path,
                                              relation_dict_path=relation_dict_path,
                                              qa_dataset_path=qa_testdataset_path)

    # ======== Model Hyper-parameters Preparation =========
    entity_embeddings = qa_dataloader.dataset.entity_embeddings
    embedding_dim = entity_embeddings.shape[-1]
    relation_embeddings = qa_dataloader.dataset.relation_embeddings
    relation_dim = relation_embeddings.shape[-1]
    vocab_size = len(qa_dataloader.dataset.word_idx)
    word_dim = 400
    hidden_dim = 350
    fc_hidden_dim = 400
    # load_from = 'answer_filtering_module/MetaQA_full_1_hop_Fri_May_12_00_03_15_2023/best_afm_model.pt'
    load_from = ''

    print(
        f"DATA: Embedding_dim: {embedding_dim}, Hidden_dim: {hidden_dim}, entity_embeddings: {entity_embeddings.shape}, Relation_Embedding: {relation_embeddings.shape}")

    # ========= Model Init =========
    model = Answer_filtering_module(entity_embeddings=entity_embeddings, embedding_dim=embedding_dim, vocab_size=vocab_size,
                                    word_dim=word_dim, hidden_dim=hidden_dim, fc_hidden_dim=fc_hidden_dim,
                                    relation_dim=relation_dim,
                                    head_bn_filepath=head_bn_path, score_bn_filepath=score_bn_path)

    # uncomment when training
    if load_from:
        # model.load_state_dict(torch.load(load_from))
        model = torch.load(load_from)

    model.to(device=torch.device('cuda'))
    model.train()

    # ======= Training Hyper-parameters Preparation ========
    TRAINING_RESULTS_DIR = os.path.join('answer_filtering_module', '_'.join([
        KG_NAME, "half" if KG_HALF else "full", str(HOPS), 'hop',
        time.asctime().replace(' ', '_').replace(':', '_')
    ]))

    if not os.path.isdir(TRAINING_RESULTS_DIR):
        os.mkdir(TRAINING_RESULTS_DIR)

    if not os.path.exists(os.path.join(TRAINING_RESULTS_DIR, 'word_idx.txt')):
        with open(os.path.join(TRAINING_RESULTS_DIR, 'word_idx.txt'), 'wt', encoding='utf-8') as outp:
            for word, idx in word_idx.items():
                outp.write(word + '\t' + str(idx) + '\n')
    best_model_path = os.path.join(TRAINING_RESULTS_DIR, 'best_afm_model.pt')
    final_model_path = os.path.join(TRAINING_RESULTS_DIR, 'final_afm_model.pt')
    N_EPOCHS = 200
    PATIENCE = 5
    LR = 0.0001
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    LR_DECAY = 0.95
    TEST_INTERVAL = 5
    TEST_TOP_K = 5
    NO_UPDATE = 0
    assert TEST_TOP_K in [1, 5, 10, 15]
    best_val_score = 0
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=adam_optimizer, gamma=LR_DECAY)

    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)
    file_logger = logging.FileHandler(os.path.join(TRAINING_RESULTS_DIR, 'training.log'), encoding='UTF-8')
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_logger)

    def correct_rate(head_entity, topK_entity_idx, answers):
        """
        :param head_entity: number index
        :param topK_entity_idx: topK list[number]
        :param answers: list[number]
        :return:
        """
        points = 0
        for candid in topK_entity_idx:
            if candid != head_entity and candid in answers:
                points += 1
        return points / len(topK_entity_idx)

    # ========== Training Step ===========
    for epoch_idx in range(N_EPOCHS):
        epoch_idx += 1
        avg_epoch_loss = 0

        print(f"Epoch: {epoch_idx} / {N_EPOCHS} - TRAIN PHASE")
        for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in tqdm(qa_dataloader):
            model.zero_grad()
            loss = model(question=batch_questions_index.cuda(),
                         questions_length=batch_questions_length.cuda(),
                         head_entity=batch_head_entity.cuda(),
                         tail_entity=batch_answers.cuda(),
                         max_sent_len=max_sent_len)
            loss.backward()
            adam_optimizer.step()
            avg_epoch_loss += loss.item()
        print(f'=> Epoch {epoch_idx}-th: Average Loss: {avg_epoch_loss / (len(qa_dataloader))}')

        if epoch_idx % TEST_INTERVAL == 0:
            model.eval()
            eval_correct_rate = 0
            print("DEV EVALUATION RUNNING")
            for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in tqdm(qa_dev_dataloader):
                ranked_topK_entity_idxs = model.get_ranked_top_k(batch_questions_index.cuda(),
                                                                 batch_questions_length.cuda(),
                                                                 batch_head_entity.cuda(), max_sent_len, K=TEST_TOP_K)
                ranked_topK_entity_idxs = ranked_topK_entity_idxs.indices.tolist()
                batch_head_entity = batch_head_entity.tolist()
                batch_correct_rate = np.sum(
                    np.array([correct_rate(head_entity=head_entity, topK_entity_idx=ranked_topK_entity_idxs[idx],
                                           answers=batch_answers[idx]) for idx, head_entity in
                              enumerate(batch_head_entity)]))
                eval_correct_rate += batch_correct_rate

            print("TEST EVALUATION RUNNING")
            for batch_questions_index, batch_questions_length, batch_head_entity, batch_answers, max_sent_len in tqdm(qa_test_dataloader):
                ranked_topK_entity_idxs = model.get_ranked_top_k(batch_questions_index.cuda(),
                                                                 batch_questions_length.cuda(),
                                                                 batch_head_entity.cuda(), max_sent_len, K=TEST_TOP_K)
                ranked_topK_entity_idxs = ranked_topK_entity_idxs.indices.tolist()
                batch_head_entity = batch_head_entity.tolist()
                batch_correct_rate = np.sum(
                    np.array([correct_rate(head_entity=head_entity, topK_entity_idx=ranked_topK_entity_idxs[idx],
                                           answers=batch_answers[idx]) for idx, head_entity in
                              enumerate(batch_head_entity)]))
                eval_correct_rate += batch_correct_rate
            eval_correct_rate /= (len(qa_test_dataloader) + len(qa_dev_dataloader))
            print(f"----------========> AVERAGE CORRECT RATE: {eval_correct_rate}")

            model.train()
            if eval_correct_rate > best_val_score + 0.0001:
                print(f'Evaluation Accuracy Hit@{TEST_TOP_K} Increases From {best_val_score} To {eval_correct_rate} => Save The Model To {best_model_path}. \n')
                # torch.save(model.state_dict(), best_model_path)
                torch.save(model, best_model_path)
                best_val_score = eval_correct_rate
                NO_UPDATE = 0
            elif NO_UPDATE >= PATIENCE:
                print(f'Model Does Not Increases In The Epoch-[{epoch_idx}], Which Has Exceed Patience.\n')
                exit(-1)
            else:
                NO_UPDATE += 1

    print(f"Final Epoch Has Reached. Stop And Save Model To {final_model_path}.")
    # torch.save(model.state_dict(), final_model_path)
    torch.save(model, final_model_path)
    print("Bingo.")
