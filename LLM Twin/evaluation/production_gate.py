import os
from openai import OpenAI

class ProductionGate:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def evaluate_llm_as_judge(self, prompt: str, generated_response: str, expected_style: str) -> float:
        evaluation_prompt = f"""
        You are an evaluator for an AI twin. 
        Evaluate the generated response based on how well it matches the expected style and factual accuracy.
        Score it from 0.0 to 1.0, where 1.0 is a perfect twin match. Output ONLY the float number.
        
        Prompt: {prompt}
        Expected Style Notes: {expected_style}
        Generated Response: {generated_response}
        """
        
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.0
        )
        try:
            return float(completion.choices[0].message.content.strip())
        except ValueError:
            return 0.0

    def run_golden_set(self, model_outputs: list[dict], threshold: float = 0.8) -> bool:
        scores = []
        for item in model_outputs:
            score = self.evaluate_llm_as_judge(
                item["prompt"], 
                item["generated"], 
                "A Learner, AI Developer, Researcher. Prefers physical books, researches autonomous EVs."
            )
            scores.append(score)
            
        avg_score = sum(scores) / len(scores)
        print(f"Average Twin Quality Score: {avg_score:.2f}")
        return avg_score >= threshold