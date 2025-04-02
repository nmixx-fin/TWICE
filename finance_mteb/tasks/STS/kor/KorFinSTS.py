from __future__ import annotations
from finance_mteb.abstasks.TaskMetadata import TaskMetadata
from ....abstasks import AbsTaskSTS


class KorFinSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="KorFinSTS",
        description="금융 텍스트에서 미묘한 의미적 변화를 탐지하여, 문장이 얼마나 유사한지 판단합니다.",
        reference="",
        dataset={
            "path": "nmixx-fin/NMIXX_kor_fin_news_STS",
            "revision": "main",
        },
        type="STS",
        category="s2s",
        eval_splits=["test"], 
        eval_langs=["kor-Hang"],
        main_score="cosine_spearman",
    )
    

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 1
        return metadata_dict
