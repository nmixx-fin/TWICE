from __future__ import annotations

from finance_mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks import AbsTaskClassification
# from ....eval_instruction import kor_task2instruction

# 절대 경로 임포트로 변경
from eval_instruction import kor_task2instruction

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

    # 전처리 코드 추가
    def dataset_transform(self):
        # 컬럼명 변경: query → text
        self.dataset = self.dataset.rename_column("query", "text")
        
        # 레이블 매핑
        labels = sorted(self.dataset["train"].unique("label_text"))
        self.dataset = self.dataset.map(
            lambda example: {
                "label": labels.index(example["label_text"]),
            },
            remove_columns=["label_text"],
        )
        self.dataset = self.dataset.class_encode_column("label")