from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


from datasets import load_dataset
import pandas as pd
import ast
from datasets import Dataset, DatasetDict

class KorOpnionHeadlineACPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="KorOpnionHeadlineACPairClassification",
        description="Financial text sentiment categorization dataset.",
        reference="",
        dataset={
            "path": "nmixx-fin/twice_kor_news_opinion_headline_pair_cls",
            "revision": "main",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="ap",
    )
    
    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")
        self.dataset = pd.DataFrame(self.dataset['train'])

        self.dataset['sentence1'] = self.dataset['sentence1'].apply(ast.literal_eval)
        self.dataset['sentence2'] = self.dataset['sentence2'].apply(ast.literal_eval)
        self.dataset['labels'] = self.dataset['labels'].apply(ast.literal_eval)

        print(f"## {len(self.dataset['sentence1'])}, {len(self.dataset['sentence2'])}, {len(self.dataset['labels'])}")

        # DataFrame을 Hugging Face Dataset으로 변환
        dataset_hf = Dataset.from_pandas(self.dataset)

        # train split만 포함한 DatasetDict 형태로 반환
        self.dataset = DatasetDict({"train": dataset_hf})
        
