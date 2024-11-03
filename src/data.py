"""
For the purpose of this project, we use only a subset of the Home Depot dataset (determined by N).  This subset is divided into a training and test set (determined by TEST_SIZE).  The training set is used to fine-tune a sentence transformer.  The test set is used to evaluate the transformer in an IR task. The test set (either the queries or the corpus) is not seen by the model in fine-tuning.

More on the dataset: https://huggingface.co/datasets/bstds/home_depot
"""
from datasets import load_dataset
import logging

SEED = 42
HF_DATASET = "bstds/home_depot"

N = 25000 # We only use N items from the Hugging Face dataset for training/testing
TEST_SIZE = 0.2
N_distractors = 200 # For evaluaton, we add some distractors (products not in the training or test set) into the corpus
RELEVANCY_THRESHOLD = 2.5 # Query/retrieval pairs with scores higher than or equal to this threshold are considered hits

def to_str(lst):
    """
    Turn a list of numbers into a list of strings
    """
    return list(map(str, lst))

def prep_eval(ds, ds_full):
    """
    Prepare a test set. Fine-tuning does not see either the queries nor the corpus.
    Return a set of queries, relevant documents, and a corpus, as required by https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html
    """
    
    #### Prepare a dictionary for queries that found relevant products,
    #### and a dictionary of the query/product association
    # Keep only the query/retrieval pairs that are highly relevant 
    ds_hits = ds["test"].filter(lambda r: r['relevance'] >= RELEVANCY_THRESHOLD)
    queries = dict(
        zip(to_str(ds_hits['id']),
            ds_hits['query']))

    logging.info("\nThe test set:")    
    logging.info(f'Number of queries: {len(queries.keys())}')

    relevant_docs = {}
    for qid, entity_id in zip(ds_hits["id"], ds_hits["entity_id"]):
        if qid not in relevant_docs:
            relevant_docs[str(qid)] = set()
        relevant_docs[str(qid)].add(str(entity_id))
    
    #### Prepare a corpus: products that are in the test set, in addition to distractors
    train_entity_id = set(ds['train']['entity_id'])
    test_entity_id = set(ds['test']['entity_id'])
    train_test_entity_id = train_entity_id.union(test_entity_id)    

    # A distractor is a product which is not in the training or the test set
    full_entity_id = set(ds_full['entity_id'])
    distractors = full_entity_id.difference(train_test_entity_id)
    distractors = list(distractors)[:N_distractors]

    corpus_entity_id = list(test_entity_id.union(distractors))

    logging.info(f'Size of corpus: {len(corpus_entity_id)} items')
    corpus = {}
    for item in ds_full:
        entity_id = item['entity_id']
        description = item['description']
        if entity_id in corpus_entity_id:
            corpus[str(entity_id)] = description
        
    return queries, relevant_docs, corpus

def prep_train_set(ds, use_score = True):
    """
    Prep a training set that follows the requirements of CosineSimilarityLoss (https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss).
    ds is a dataset, not a dictionary of dataset.
    If use_score is True, keep the score (but normalised to [0,1]). Otherwise, keep only sentence pairs with score > 2.5. 
    """
    if use_score:
        train_samples = ds.select_columns(['query', 'description', 'relevance'])
        train_samples = train_samples.rename_column('query', 'sentence_A')
        train_samples = train_samples.rename_column('description', 'sentence_B')
        train_samples = train_samples.rename_column('relevance', 'score')

        # Scale the score from [1, 3]to [0.0, 1.0], as required by the loss function
        def scale_item(item):
            item['score'] = (item['score'] - 1.0) / 2.0
            return item
        train_samples = train_samples.map(scale_item)
    else:
        train_samples = ds.filter(lambda item: item['relevance'] > 2.5)
        train_samples = train_samples.select_columns(["query", "description"])
        train_samples = train_samples.rename_column('query', 'anchor')
        train_samples = train_samples.rename_column('description', 'positive')
    
    logging.info(f'\nTraining set size: {train_samples.shape[0]}')
    print('\nTraining set size:', train_samples.shape[0])

    return train_samples

def print_testset_samples(test_queries, test_relevant_docs, test_corpus, n_items = 2):
    """
    Print some sample queries and retrievals to check that the test set is ok.
    """
    print("\n###########################")
    print("Sample items in the test set")
    for q_id in list(test_queries.keys())[:n_items]:
        print("\n[QUERY]:", test_queries[q_id])

        # test_relevant_docs[q_id] is a set. After turning to a list, take the first one (the set always has 1 element)
        entity_id = list(test_relevant_docs[q_id])[0]
        print("[RETRIEVED]:", test_corpus[entity_id])

def prep_data(use_score = True):
    """
    Load the Hugging Face dataset and prepare train/test set.
    If use_score is True, include the score. Otherwise, only return sentence pairs with high scores
    """
    logging.info("Loading data...")
    ds_full = load_dataset("bstds/home_depot", split = "train")
    ds_full = ds_full.shuffle(seed = SEED)

    # Take a subset of the loaded data and split it
    ds = ds_full.select(range(N))    
    ds = ds.train_test_split(test_size = TEST_SIZE, seed = SEED)

    ## Prep the training set. Each sample consists of two texts and a score (float, from 1.0 to 3.0)
    train_samples = prep_train_set(ds['train'], use_score = use_score)
    
    ## Restructure the test set for IR evaluation
    test_queries, test_relevant_docs, test_corpus = prep_eval(ds, ds_full)

    print_testset_samples(test_queries, test_relevant_docs, test_corpus)
    
    return test_queries, test_relevant_docs, test_corpus, train_samples

if __name__ == '__main__':
    test_queries, test_relevant_docs, test_corpus, train_samples = prep_data(use_score = False)

    print('**** Examples of the test set:')
    print_testset_samples(test_queries, test_relevant_docs, test_corpus, n_items = 10)
    
