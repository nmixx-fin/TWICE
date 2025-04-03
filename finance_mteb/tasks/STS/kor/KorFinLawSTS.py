from __future__ import annotations
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks import AbsTaskSTS

class KorFinLawSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorFinLawSTS",
        description="금융 법률 텍스트에서 미묘한 의미적 변화를 탐지하여, 문장이 얼마나 유사한지 판단합니다.",
        reference="",
        dataset={
            "path": "nmixx-fin/NMIXX_kor_fin_law_STS",
            "revision": "main",
        },
        type="STS",
        category="s2s",
        eval_splits=["train"], 
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
    )

    def dataset_transform(self):
        self.dataset = self.dataset.rename_columns({
            "text1": "sentence1",
            "text2": "sentence2",
        })
        
        def validate_example(example):
            """모든 문장을 문자열로 강제 변환 (NaN 방지)"""
            example['sentence1'] = str(example['sentence1']) if example['sentence1'] is not None else ""
            example['sentence2'] = str(example['sentence2']) if example['sentence2'] is not None else ""
            return example
            
        self.dataset = self.dataset.map(validate_example)

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict.update({
            "min_score": 0,
            "max_score": 1,
            "text_columns": ["sentence1", "sentence2"]
        })
        return metadata_dict