from __future__ import annotations

from finance_mteb.abstasks.AbsTaskSummarization import AbsTaskSummarization
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
import pandas as pd
from datasets import Dataset, DatasetDict


class KorOpinionSummarization(AbsTaskSummarization):
    metadata = TaskMetadata(
        name="KorOpinionSummarization",
        description="주어진 칼럼에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
        reference="",
        dataset={
            "path": "nmixx-fin/twice_kr_finance_column_summ",
            "revision": "main",
        },
        type="Summarization",
        category="p2p",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
    )

    def dataset_transform(self):
        # 데이터셋을 DataFrame으로 변환
        self.dataset = pd.DataFrame(self.dataset['train'])
        
        # 컬럼명 변경
        if 'label' in self.dataset.columns:
            self.dataset = self.dataset.rename(columns={'label': 'score'})
        if 'sentence' in self.dataset.columns:
            self.dataset = self.dataset.rename(columns={'sentence': 'text'})
        
        # DataFrame을 Hugging Face Dataset으로 변환
        dataset_hf = Dataset.from_pandas(self.dataset)
        
        # train split만 포함한 DatasetDict 형태로 반환
        self.dataset = DatasetDict({"train": dataset_hf})

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict