from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction

class KorFinascSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinascSentClassification",
        description=kor_task2instruction["KorFinascSentClassification"],
        reference="amphora/korfin-asc",
        dataset={
            "path": "nmixx-fin/twice_korfin-asc_sent_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )