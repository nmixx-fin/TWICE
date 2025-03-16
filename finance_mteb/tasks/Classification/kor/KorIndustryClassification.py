from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
# from ....eval_instruction import kor_task2instruction

# 절대 경로 임포트로 변경
from eval_instruction import kor_task2instruction

class KorIndustryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorIndustryClassification",
        description=kor_task2instruction["KorIndustryClassification"],
        reference="Industry/Investment Analysis Report",
        dataset={
            "path": "nmixx-fin/twice_kr_industry_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )
    
    # huggingface 데이터셋에 이미 text 칼럼으로 되어 있어서 칼럼 이름 바꿀 필요 없음
    # def dataset_transform(self):
    #     self.dataset = self.dataset.rename_column("sentence", "text")