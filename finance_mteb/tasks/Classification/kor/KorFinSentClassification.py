from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
from ....eval_instruction import kor_task2instruction

class KorFinSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinSentClassification",
        description=kor_task2instruction["KorFinSentClassification"],
        reference="Financial News (Naver News)",
        dataset={
            "path": "nmixx-fin/twice_kr_fin_news_sent_cls",
            "revision": "main",
        },
        type="Classification",
        category="s2s",
        eval_splits=["train"],
        eval_langs=["kor-Hang"],
        main_score="accuracy",
    )