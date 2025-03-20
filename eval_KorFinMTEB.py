import os

import logging
import argparse
from example_models.flag_dres_model import FlagDRESModel
from example_models.flag_icl_Model import FLAGICLModel
from example_models.openai_embed_model import OpenAIEmbedder
from example_models.e5_mistral_model import E5DRESModel
from finance_mteb import MTEB
from sentence_transformers import SentenceTransformer
from example_models.gte_model import GTERESModel
from eval_instruction import kor_task2instruction

TASK_LIST_CLASSIFICATION = [
    "KorFinSentClassification",
    "KorESGClassification",
    "KorFOMCClassification",
    "KorIndustryClassification",
    "KorFinascSentClassification",
    "KorFinMMLUClassification",
    "KorFinBQAClassification",
    "KorFinMCQAClassification",
    "KorNewsBQAClassification"
]

TASK_LIST_RETRIEVAL = [
    #"TATQARetrieval",
    "KorTATQARetrieval",
    # "USNewsRetrieval",
    "KorNewsRetrieval",
    # "TheGoldmanEnRetrieval",
    "KorBoKDictRetrieval",
    "KorFSSDictRetrieval",
    "KorMarketReportRetrieval"
]

TASK_LIST_CLUSTERING = [
    # "WikiCompany2IndustryClustering",
    "KorDartCompany2IndustryClustering",
    "KorMinDS14Clustering"
]

TASK_LIST_RERANKING = [
    #  "FiQA2018Reranking",
     "KorFiQA2018Reranking"
]

TASK_LIST_STS = [
    # "FinSTS",
    "KorFinSTS"
]

TASK_LIST_PAIRCLASSIFICATION = [
     # "HeadlineACPairClassification",
     "KorHeadlineACPairClassification",
     "KorContentACPairClassification",
     "KorOpnionHeadlineACPairClassification"
]

TASK_LIST_SUMMARIZATION = [
    "KorLawSummarization",
    "KorNewsSummarization",
    "KorOpinionSummarization",
    "KorFinanceNewsSummarization",
    "KorFinanceColumnSummarization"
] 

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-large-zh", type=str)
    parser.add_argument('--task_type', default=None, type=str)
    parser.add_argument('--add_instruction', action='store_true', help="whether to add instruction for query")
    parser.add_argument('--pooling_method', default='cls', type=str) # model마다 pooling method

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    if 'bge' in args.model_name_or_path and not 'icl' in args.model_name_or_path:
        model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval=None,
                            pooling_method='cls')
    elif 'KURE' in args.model_name_or_path and not 'icl' in args.model_name_or_path:
        model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval=None,
                            pooling_method='cls')
                            
    elif 'icl' in args.model_name_or_path:
        model = FLAGICLModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다..")
    elif 'text-embedding' in args.model_name_or_path:
        model = OpenAIEmbedder(engine=args.model_name_or_path)

    elif 'gte' in args.model_name_or_path or 'stella' in args.model_name_or_path:
        model = GTERESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다.",
                            pooling_method="last")
    elif 'e5-mistral' in args.model_name_or_path:
        model = E5DRESModel(model_name_or_path=args.model_name_or_path,
                    query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다.",
                    pooling_method="last")
    else:
        model = SentenceTransformer(args.model_name_or_path, trust_remote_code=True)
        model.max_seq_length = 256

    RUNNING_TASK = []

    if args.task_type =="CLASSIFICATION":
        RUNNING_TASK += TASK_LIST_CLASSIFICATION
    if args.task_type =="RETRIEVAL":
        RUNNING_TASK += TASK_LIST_RETRIEVAL
    if args.task_type =="CLUSTERING":
        RUNNING_TASK += TASK_LIST_CLUSTERING
    if args.task_type =="RERANKING":
        RUNNING_TASK += TASK_LIST_RERANKING
    if args.task_type =="STS":
        RUNNING_TASK += TASK_LIST_STS
    if args.task_type =="SUMM":
        RUNNING_TASK += TASK_LIST_SUMMARIZATION
    if args.task_type =="PAIRCLASSIFICATION":
        RUNNING_TASK += TASK_LIST_PAIRCLASSIFICATION

    for task in RUNNING_TASK:
        logger.info(f"Running task: {task}")
        if task in kor_task2instruction.keys():
            instruction = kor_task2instruction[task]
            if hasattr(model, 'set_prompt'):
                model.set_prompt(instruction)
                logger.info(f'Setting Prompt: {instruction} For Task: {task}')
            elif hasattr(model, 'query_instruction_for_retrieval'):
                model.query_instruction_for_retrieval = instruction
                logger.info(f'Setting Query Instruction: {instruction} For Task: {task}')

        evaluation = MTEB(tasks=[task])
        logger.info('Running evaluation for task: {}'.format(evaluation))
        evaluation.run(model, output_folder=f"results/{args.model_name_or_path.split('/')[-1]}",encode_kwargs={"batch_size": 64})