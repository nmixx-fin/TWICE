from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

# from ....eval_instruction import kor_task2instruction

# 절대 경로 임포트로 변경
from eval_instruction import kor_task2instruction

class KorESGClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorESGClassification",
        description=kor_task2instruction["KorESGClassification"],
        reference="",
        dataset={
            "path": "nmixx-fin/twice_kr_esg_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )