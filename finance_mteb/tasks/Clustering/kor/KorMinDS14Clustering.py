
from __future__ import annotations

from finance_mteb.abstasks.AbsTaskClustering import AbsTaskClustering
from finance_mteb.abstasks.AbsTaskClusteringFast import clustering_downsample
from finance_mteb.abstasks.TaskMetadata import TaskMetadata


class KorMInDS14Clustering(AbsTaskClustering):
    metadata = TaskMetadata(
        name="KorMInDS14Clustering",
        description="사업 보고서에 나와있는 기업 설명 텍스트를 기반으로 산업군을 분류합니다.",
        reference="https://arxiv.org/pdf/2104.08524",
        dataset={
            "path": "nmixx-fin/twice_kr_minds14_clustering",
            "revision": "main",
        },
        type="Clustering",
        category="p2p",
        eval_splits=["test"],
        eval_langs=["kor-Hang"],
        main_score="v_measure",
    )
