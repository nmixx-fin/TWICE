from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction

class KorFinMMLUClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinMMLUClassification",
        description=kor_task2instruction["KorFinMMLUClassification"],
        reference="allganize/financial-mmlu-ko",
        dataset={
            "path": "nmixx-fin/twice_kr_financial_mmlu_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )