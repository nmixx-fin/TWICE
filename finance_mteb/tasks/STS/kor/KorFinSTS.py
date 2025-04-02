from __future__ import annotations
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks import AbsTaskSTS
from ....evaluation.evaluators import STSEvaluator
from ....MTEBResults import ScoresDict
import numpy as np

class KorFinSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorFinSTS",
        description="금융 텍스트에서 미묘한 의미적 변화를 탐지하여, 문장이 얼마나 유사한지 판단합니다.",
        reference="",
        dataset={
            "path": "nmixx-fin/NMIXX_kor_fin_news_STS",
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
        
        # score 값들의 고유값 확인
        unique_scores = np.unique(self.dataset['score'])
        
        # 0과 1만 있는지 확인 (float나 int 모두 고려)
        is_binary = all(np.isclose(score, 0) or np.isclose(score, 1) for score in unique_scores)
        
        if is_binary:
            # 이진 분류인 경우 int로 변환
            self.dataset = self.dataset.map(lambda x: {**x, 'score': int(x['score'])}, remove_columns=['score'])
            self.is_binary = True
        else:
            # 이진 분류가 아닌 경우 float 유지
            self.is_binary = False

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict.update({
            "min_score": 0,
            "max_score": 1,
            "text_columns": ["sentence1", "sentence2"]
        })
        return metadata_dict

    def _evaluate_subset(self, model, data_split, **kwargs) -> ScoresDict:
        if self.is_binary:
            # 이진 분류인 경우 정규화 없이 원래 값 사용
            evaluator = STSEvaluator(
                data_split["sentence1"],
                data_split["sentence2"],
                data_split["score"],
                task_name=self.metadata.name,
                **kwargs,
            )
        else:
            # 이진 분류가 아닌 경우 정규화 수행
            def normalize(x):
                return (x - self.min_score) / (self.max_score - self.min_score)
            normalized_scores = list(map(normalize, data_split["score"]))
            evaluator = STSEvaluator(
                data_split["sentence1"],
                data_split["sentence2"],
                normalized_scores,
                task_name=self.metadata.name,
                **kwargs,
            )
        
        scores = evaluator(model)
        self._add_main_score(scores)
        return scores