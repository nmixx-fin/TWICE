
from __future__ import annotations

from finance_mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from finance_mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
import pandas as pd
import ast
from datasets import load_dataset, Dataset, DatasetDict


class KorFinAscCompany2IndustryClustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="KorFinAscCompany2IndustryClustering",
        description="Clustering the related industry domain according to the company description.",
        reference="https://aclanthology.org/W18-6532.pdf",
        dataset={
            "path": "nmixx-fin/twice_dart_company2industry_clustering",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
    )
    
    def dataset_transform(self):
        self.dataset = pd.DataFrame(self.dataset['test'])

        self.dataset['sentences'] = self.dataset['sentences'].apply(ast.literal_eval)
        self.dataset['labels'] = self.dataset['labels'].apply(ast.literal_eval)

        # DataFrame을 Hugging Face Dataset으로 변환
        dataset_hf = Dataset.from_pandas(self.dataset)

        # train split만 포함한 DatasetDict 형태로 반환
        self.dataset = DatasetDict({"test": dataset_hf})