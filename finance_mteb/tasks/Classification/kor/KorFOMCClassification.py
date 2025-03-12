from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification


class KorFOMCClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFOMCClassification",
        description="FOMC에서 제공된 금융 텍스트를  'Hawkish', 'Dovish', 'Neutral' 클래스 중 하나로 분류합니다.",
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
    def dataset_transform(self):
        self.dataset = self.dataset.rename_column("sentence", "text")