from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

class KorFinSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinSentClassification",
        description="주어진 금융 텍스트의 감성을 긍정, 부정, 중립 중 하나로 분류합니다.",
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