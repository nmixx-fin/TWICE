# Instruction List For Causual Embedding Model

task2instruction = {
    "FinancialPhraseBankClassification": "Classify the sentiment of a given finance text as either positive, negative, or neutral.",
    "FinSentClassification": "Classify the sentiment of a given finance text as either positive, negative, or neutral.",
    "FiQAClassification": "Perform aspect based financial sentiment classification.",
    "SemEva2017Classification": "Classify the sentiment of a given finance text as either positive, negative, or neutral.",
    "FLSClassification": "Classify the sentence into 'not-fls', 'specific fls', or 'non-specific fls' class.",
    "ESGClassification": "Classify the following sentence into one of the 'environmental','social', 'governance', 'non-esg' classes.",
    "FOMCClassification": "Classify the following sentence from FOMC into 'hawkish', 'dovish', or 'neutral' class.",
    "FinancialFraudClassification": "Detecting financial fraud from the given text.",
    "FinNSPClassification": "Perform financial sentiment classification.",
    "FinChinaSentimentClassification": "Classify the sentiment of a given finance text as either positive, negative, or neutral.",
    "FinFEClassification": "Classify the sentiment of a given financial social media text.",
    "OpenFinDataSentimentClassification": "Classify the sentiment of a given finance text.",
    "Weibo21Classification": "Classify fake news from the given text.",

    "FiQA2018Retrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "FinanceBenchRetrieval": "Given a financial question, retrieve the related context.",
    "HC3Retrieval": "Given a financial question, retrieve relevant passages that answer the query.",
    "Apple10KRetrieval": "Given a financial question, retrieve the related context.",
    "FinQARetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "TATQARetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "USNewsRetrieval": "Given a financial question, retrieve documents that can help answer the question.",
    "TradeTheEventEncyclopediaRetrieval": "Given a financial term, retrieve the related context.",
    "TradeTheEventNewsRetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "TheGoldmanEnRetrieval": "Given a financial term, retrieve the related context.",

    "FinTruthQARetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "FinEvaRetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "AlphaFinRetrieval": "Given a financial question, retrieve user replies that best answer the question.",
    "DISCFinLLMRetrieval": "Given a financial question, retrieve documents that answer the query.",
    "DISCFinLLMComputingRetrieval": "Given a financial question, retrieve the best answer.",
    "DuEEFinRetrieval": "Given a financial question, retrieve documents that can help answer the question.",
    "SmoothNLPRetrieval": "Given a financial question, retrieve documents that can help answer the question.",
    "THUCNewsRetrieval": "Given a financial question, retrieve documents that can help answer the question.",
    "FinEvaEncyclopediaRetrieval": "Given a financial term, retrieve the related context.",
    "TheGoldmanZhRetrieval": "Given a financial term, retrieve the related context.",

    "MInDS14EnClustering": "Identify the main category of the intention for the given text.",
    "ComplaintsClustering": "Identify the main category of the consumer complaint.",
    "PiiClustering": "Cluster the given text based on the personally identifiable information.",
    "FinanceArxivS2SClustering": "Identify the main category of finance papers based on the titles",
    "FinanceArxivP2PClustering": "Identify the main category of finance papers based on the abstracts.",
    "WikiCompany2IndustryClustering": "Identify industries from company descriptions.",

    "MInDS14ZhClustering": "Identify the main category of the intention for the given text.",
    "FinNLClustering": "Identify the main category of the given finance news.",
    "CCKS2022Clustering": "Identify the main event of the given text.",
    "CCKS2020Clustering": "Identify the main event of the given text.",
    "CCKS2019Clustering": "Identify the main event of the given text.",

    "FinFactReranking": "Given a financial question, retrieve documents that answer the query.",
    "FiQA2018Reranking": "Given a financial question, retrieve documents that can help answer the question.",
    "HC3Reranking": "Given a financial question, retrieve relevant passages that answer the query.",
    "FinEvaReranking": "Given a financial question, retrieve user replies that best answer the question.",
    "DISCFinLLMReranking": "Given a financial query, retrieve the related context.",

    "FinSTS": "Detecting Subtle Semantic Shifts in Financial Narratives.",
    "FINAL": "Retrieve semantically similar finance text.",
    "AFQMC": "Retrieve semantically similar finance text.",
    "BQCorpus": "Retrieve semantically similar finance text.",

    "Ectsum": "Given a news text, retrieve other semantically similar summaries",
    "FINDsum": "Given a finance document, retrieve other semantically similar summaries",
    "FNS2022sum": "Given a 10K document, retrieve other semantically similar summaries",
    "FiNNAsum": "Given a news text, retrieve other semantically similar summaries",
    "FinEvaHeadlinesum": "Given a finance document, retrieve other semantically similar summaries",
    "FinEvasum": "Given a finance document, retrieve other semantically similar summaries",

    "HeadlineACPairClassification": "Classify the sentiment of a given finance text.",
    "HeadlinePDDPairClassification": "Classify the sentiment of a given finance text.",
    "HeadlinePDUPairClassification": "Classify the sentiment of a given finance text.",
    "AFQMCPairClassification": "Matching the semantically similar questions.",
}

# The Korean translation of the original instruction
kor_task2instruction = {
    # Classification
    "KorFinSentClassification": "주어진 금융 텍스트의 감성을 긍정, 부정, 중립 중 하나로 분류합니다.",
    "KorESGClassification": "주어진 금융 텍스트를 'E', 'S', 'G', 'Non-ESG' 클래스 중 하나로 분류합니다.",
    "KorFOMCClassification": "FOMC에서 제공된 금융 텍스트를  'Hawkish', 'Dovish', 'Neutral' 클래스 중 하나로 분류합니다.",
    "KorIndustryClassification" : "산업 분석 리포트 내 텍스트를 보고, 어떤 산업군에 대한 서술인지 ['건설', '반도체', '석유화학', '유통', '은행', '음식료', '자동차', '조선', '철강금속', '통신'] 클래스 중 하나로 분류합니다.",
    "KorFinascSentClassification" : "금융 분석 보고서 내의 텍스트를 보고, 각 aspect에 따른 src의 감정을 'POSITIVE', 'NEUTRAL', 'NEGATIVE' 클래스 중 하나로 분류합니다.",
    "KorFinMMLUClassification" : "주어진 금융 도메인의 질문과 선택지가 주어졌을 때, 제공된 선택지 1, 2, 3, 4, 5 중 질문에 대한 정답 선택지 번호를 반환합니다.",
    "KorFinBQAClassification" : "주어진 금융 도메인 텍스트를 보고, 질문에 대하여 해당 텍스트가 올바른지 0 (False), 1 (True) 로 분류합니다.",
    "KorFinMCQAClassification" : "주어진 금융 도메인 텍스트를 보고, 질문과 선택지가 주어졌을 때, 제공된 선택지 1, 2, 3, 4 중 질문에 대한 정답 선택지 번호를 반환합니다.",
    "KorNewsBQAClassification" : "주어진 금융 도메인 텍스트를 보고, 질문에 대하여 해당 텍스트가 올바른지 0 (False), 1 (True) 중 하나로 분류합니다.",

    # Retrieval
    "KorTATQARetrieval": "재무 리포트에서 주어진 금융 텍스트에 대해 가장 적절한 사용자 답변을 검색합니다.",
    "KorNewsRetrieval": "주어진 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorBoKDictRetrieval": "주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorFSSDictRetrieval": "주어진 금융 용어에 대해 답변에 도움이 될 문서를 검색합니다.",
    "KorMarketReportRetrieval" : "주식 시장과 관련된 금융 텍스트에 대해 답변에 도움이 될 문서를 검색합니다.",

    # Clustering
    "KorMInDS14EnClustering": "주어진 텍스트의 주요 의도를 분류합니다.",
    "KorDartCompany2IndustryClustering" : "사업 보고서에 나와있는 기업 설명 텍스트를 기반으로 산업군을 분류합니다.",

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
