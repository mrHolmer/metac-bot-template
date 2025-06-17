import argparse
import asyncio
import logging
from datetime import datetime
from typing import Literal
import wikipediaapi
import requests  # For NewsAPI free tier

from forecasting_tools import (
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    clean_indents,
)

logger = logging.getLogger(__name__)

class FreeForecaster(ForecastBot):
    _max_concurrent_questions = 1  # Conservative for local LLMs
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wiki = wikipediaapi.Wikipedia('en')
        self.newsapi_key = None  # Set NEWS_API_KEY in env for news

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Free research using Wikipedia and NewsAPI free tier"""
        research = []
        
        # 1. Wikipedia research
        wiki_page = self.wiki.page(question.question_text[:50])
        if wiki_page.exists():
            research.append(f"Wikipedia Summary:\n{wiki_page.summary[:1000]}")
        
        # 2. NewsAPI (free tier - 100 requests/day)
        if os.getenv('NEWS_API_KEY'):
            url = f"https://newsapi.org/v2/everything?q={question.question_text[:50]}&apiKey={os.getenv('NEWS_API_KEY')}"
            response = requests.get(url)
            if response.status_code == 200:
                articles = response.json().get('articles', [])[:3]  # Limit to 3 articles
                research.append("News Headlines:\n" + "\n".join(
                    f"- {a['title']} ({a['url']})" for a in articles
                ))
        
        return "\n\n".join(research) if research else "No free research available"

    async def _get_free_llm(self):
        """Connect to local Ollama/LM Studio instance"""
        return GeneralLlm(
            model="http://localhost:11434",  # Ollama default
            # model="http://localhost:1234",  # LM Studio default
            temperature=0.3,
            timeout=60,
            model_name="mistral"  # Free local model
        )

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        llm = await self._get_free_llm()
        prompt = clean_indents(f"""
        [Simplified free version]
        Question: {question.question_text}
        Current Date: {datetime.now().strftime("%Y-%m-%d")}
        Research: {research or "No research conducted"}
        
        Analyze and give:
        1. Key factors affecting outcome
        2. Most likely outcome (0-100% probability)
        3. Final answer format: "Probability: XX%"
        """)
        reasoning = await llm.invoke(prompt)
        prediction = PredictionExtractor.extract_last_percentage_value(reasoning)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    # Similar simplified versions for numeric/multiple choice...
    # (Copy structure from original but use _get_free_llm())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    free_bot = FreeForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=3,  # Reduced for free tier
        publish_reports_to_metaculus=True,
        skip_previously_forecasted_questions=True,
    )
    
    asyncio.run(free_bot.forecast_on_tournament(MetaculusApi.CURRENT_AI_COMPETITION_ID))
