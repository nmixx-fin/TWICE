class KorIndustryClassification(AbsTaskClassification):
    metadata = TaskMetadata(
        name="KorIndustryClassification",
        description="산업 분석 리포트 내 텍스트를 보고, 어떤 산업군에 대한 서술인지 ['건설', '반도체', '석유화학', '유통', '은행', '음식료', '자동차', '조선', '철강금속', '통신'] 클래스 중 하나로 분류합니다.",
        dataset={
            "path": "nmixx-fin/twice_kr_industry_cls",
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