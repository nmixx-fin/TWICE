from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskPairClassification import AbsTaskPairClassification


from datasets import load_dataset
import pandas as pd
import ast
from datasets import Dataset, DatasetDict


class HeadlineACPairClassification(AbsTaskPairClassification):
    metadata = TaskMetadata(
        name="HeadlineACPairClassification",
        description="Financial text sentiment categorization dataset.",
        reference="",
        dataset={
            "path": "nmixx-fin/twice_ko-trans-HeadlineAC-PairClassification",
            "revision": "main",
        },
        type="PairClassification",
        category="s2s",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="ap",
    )

    def dataset_transform(self):
        # 컬럼 이름 변경
        self.dataset = self.dataset.rename_column("sent1", "sentence1")
        self.dataset = self.dataset.rename_column("sent2", "sentence2")

