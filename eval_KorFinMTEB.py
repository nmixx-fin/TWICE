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
import sys
from typing import List, Optional
import numpy as np
import torch

# 로깅 설정 추가
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# finance_mteb 관련 모듈의 로그 레벨을 INFO로 설정
logging.getLogger('finance_mteb').setLevel(logging.INFO)

TASK_LIST_CLASSIFICATION = [
    "KorFinSentClassification",
    "KorESGClassification",
    "KorFOMCClassification",
    "KorIndustryClassification",
    "KorFinascSentClassification",
    "KorFinMMLUClassification",
    "KorFinBQAClassification",
    "KorFinMCQAClassification",
    "KorNewsBQAClassifcation"
]

TASK_LIST_RETRIEVAL = [
    # "TATQARetrieval",
    # "USNewsRetrieval",
    # "TheGoldmanEnRetrieval",
    "KorBoKDictRetrieval",
    "KorFSSDictRetrieval",
    "KorTATQARetrieval",
    "KorNewsRetrieval",
    "KorMarketReportRetrieval"
]

TASK_LIST_CLUSTERING = [
    # "WikiCompany2IndustryClustering",
    "KorDartCompany2IndustryClustering",
    "KorMInDS14Clustering"
]

TASK_LIST_RERANKING = [
    #  "FiQA2018Reranking",
     "KorFiQA2018Reranking"
]

TASK_LIST_STS = [
    # "FinSTS"
    "KorFinSTS",
    # "KorDartReportSTS",
    # "KorFinLawSTS",
    # "KorFinReportSTS",
]

TASK_LIST_PAIRCLASSIFICATION = [
     # "HeadlineACPairClassification",
     "KorHeadlineACPairClassification",
     "KorContentACPairClassification",
     "KorOpnionHeadlineACPairClassification"
]

TASK_LIST_SUMMARIZATION = [
    "KorFinanceColumnSummarization",
    "KorFinanceNewsSummarization",
    "KorLawSummarization",
    "KorNewsSummarization",
    "KorOpinionSummarization",
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
    # bge-icl
    if 'icl' in args.model_name_or_path:
        model = FLAGICLModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다..")
    
    # bge-m3, kure
    elif any(keyword in args.model_name_or_path.lower() for keyword in ['bge', 'kure']) and not 'icl' in args.model_name_or_path:
        model = FlagDRESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval=None,
                            pooling_method='cls')
    
    # openai
    elif 'text-embedding' in args.model_name_or_path:
        model = OpenAIEmbedder(engine=args.model_name_or_path)

    # qwen
    elif 'gte' in args.model_name_or_path or 'stella' in args.model_name_or_path or 'qwen' in args.model_name_or_path:
        model = GTERESModel(model_name_or_path=args.model_name_or_path,
                            query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다.",
                            pooling_method="last")
    
    # e5-mistral
    elif 'e5-mistral' in args.model_name_or_path or 'nmixx-e5' in args.model_name_or_path:
        model = E5DRESModel(model_name_or_path=args.model_name_or_path,
                    query_instruction_for_retrieval="주어진 웹 검색 쿼리에 대해 관련된 문서를 검색하여 답을 제공합니다.",
                    pooling_method="last")
    
    # minilm, instructor
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
    if args.task_type =="SUMMARIZATION":
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