class KorFinascSentClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorFinascSentClassification",
        description="주어진 금융 텍스트의 감성을 긍정, 부정, 중립 중 하나로 분류합니다.",
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