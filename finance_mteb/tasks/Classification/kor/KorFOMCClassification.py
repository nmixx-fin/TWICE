from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
# from ....eval_instruction import kor_task2instruction

# 절대 경로 임포트로 변경
from eval_instruction import kor_task2instruction

class KorFOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFOMCClassification",
        description=kor_task2instruction["KorFOMCClassification"],
        reference="Financial News (Naver News)",
        dataset={
            "path": "nmixx-fin/twice_kr_fomc_cls",
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