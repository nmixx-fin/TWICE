from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification

class KorESGClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorESGClassification",
        description="주어진 금융 텍스트를 'E', 'S', 'G', 'Non-ESG' 클래스 중 하나로 분류합니다.",
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