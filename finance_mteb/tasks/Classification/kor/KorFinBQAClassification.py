from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction


class KorFinBQAClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinBQAClassification",
        description=kor_task2instruction["KorFinBQAClassification"],
        reference="FINNUMBER/QA_Instruction",
        dataset={
            "path": "nmixx-fin/twice_kr_financial_bqa_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )