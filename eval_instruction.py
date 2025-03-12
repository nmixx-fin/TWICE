# Instruction List For Causual Embedding Model

# The Korean translation of the original instruction
kor_task2instruction = {
    # Classification
    "KorFinSentClassification": "주어진 금융 텍스트의 감성을 긍정, 부정, 중립 중 하나로 분류합니다.",
    "KorESGClassification": "주어진 금융 텍스트를 'E', 'S', 'G', 'Non-ESG' 클래스 중 하나로 분류합니다.",
    "KorFOMCClassification": "FOMC에서 제공된 금융 텍스트를  'Hawkish', 'Dovish', 'Neutral' 클래스 중 하나로 분류합니다.",
    "KorIndustryClassification" : "산업 분석 리포트 내 텍스트를 보고, 어떤 산업군에 대한 서술인지 ['건설', '반도체', '석유화학', '유통', '은행', '음식료', '자동차', '조선', '철강금속', '통신'] 클래스 중 하나로 분류합니다."
    "KorFinascSentClassification" : "금융 분석 보고서 내의 텍스트를 보고, 각 aspect에 따른 src의 감정을 'POSITIVE', 'NEUTRAL', 'NEGATIVE' 클래스 중 하나로 분류합니다.",
    "KorFinancialMMLUClassification" : "금융 도메인의 질문과 선택지가 주어졌을 때, 제공된 선택지 중 하나의 답변을 반환합니다.",
    "KorFinancialBQAClassification" : "주어진 금융 도메인 질문에 대해, 해당 텍스트가 올바른지 'Yes', 'No' 중 하나로 분류합니다.",
    "KorFinancialMCQAClassification" : "금융 도메인의 질문과 선택지가 주어졌을 때, 제공된 선택지 중 하나의 답변을 반환합니다.",
    "KorNewsBQAClassification" : "주어진 금융 도메인 텍스트에 대해, 해당 텍스트가 올바른지 'Yes', 'No' 중 하나로 분류합니다.",

    # Retrieval
    "KorTATQARetrieval": "재무 리포트에서 주어진 금융 텍스트에 대해 가장 적절한 사용자 답변을 검색합니다.",
    "KorNewsRetrieval": "주어진 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorBoKDictRetrieval": "주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorFSSDictRetrieval": "주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorMarketReportRetrieval" : "주식 시장과 관련된 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",

    # Clustering
    "KorMInDS14EnClustering": "주어진 텍스트의 주요 의도를 분류합니다.",
    "KorDartCompany2IndustryClustering" : "사업 보고서에 나와있는 기업 설명 텍스트를 기반으로 산업군을 분류합니다."

    # Reranking
    "KorFiQA2018Reranking": "주어진 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",

    # STS
    "KorFinSTS": "금융 텍스트에서 미묘한 의미적 변화를 탐지하여, 문장이 얼마나 유사한지 판단합니다.",

    # PairCLS
    "KorHeadlineACPairClassification": "주어진 금융 뉴스 헤드라인의 감성을 분류합니다.",
    "KorContentACPairClassification": "주어진 금융 뉴스 본문의 감성을 분류합니다.",
    "KorOpnionHeadlineACPairClassification": "주어진 금융 칼럼 헤드라인의 감성을 분류합니다.",

    # Summ
    "KorLawSummarization" : "주어진 법률에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
    "KorNewsSummarization": "주어진 뉴스에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
    "KorOpinionSummarization": "주어진 칼럼에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
    "KorFinanceNewsSummarization": "주어진 경제 뉴스에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
    "KorFinanceColumnSummarization": "주어진 경제 칼럼에 대한 요약문이 맞는지 0 (False), 1 (True) 로 분류합니다.",
}
