from .base import Probe
import re

class FactualityProbe(Probe):
    """Probe to test LLMs' factual accuracy."""
    def __init__(self, name="FactualityProbe"):
        super().__init__(name)

    def run(self, model, factual_questions: list, **kwargs):
        """Run the factuality probe on the given model with a list of factual questions.
        
        Args:
            model: The model to test
            factual_questions: A list of dictionaries, each containing:
                - 'question': The factual question to ask
                - 'reference_answer': The correct answer for evaluation
                - 'category' (optional): Category of the question (e.g., 'history', 'science')
        
        Returns:
            A list of dictionaries, each containing:
                - 'question': The original question
                - 'reference_answer': The reference answer
                - 'model_answer': The model's response
                - 'category': The question category (if provided)
                - 'evaluation': Evaluation results (if evaluator is provided)
        """
        results = []
        
        for item in factual_questions:
            question = item['question']
            reference = item['reference_answer']
            category = item.get('category', 'general')
            
            # Format the prompt to encourage factual responses
            prompt = f"Please answer this factual question accurately and concisely: {question}"
            
            # Get model response
            model_answer = model.generate(prompt, **kwargs)
            
            # Basic evaluation - extract the direct answer if possible
            # This is a simple implementation; more sophisticated evaluation could be added
            extracted_answer = self._extract_direct_answer(model_answer)
            
            result = {
                'question': question,
                'reference_answer': reference,
                'model_answer': model_answer,
                'extracted_answer': extracted_answer,
                'category': category
            }
            
            results.append(result)
            
        return results
    
    def _extract_direct_answer(self, response):
        """Extract a direct answer from a longer response if possible."""
        # Look for patterns like "The answer is X" or "X is the answer"
        patterns = [
            r"(?:the answer is|the correct answer is|answer:)\s*(.*?)(?:\.|$)",
            r"(?:^|\n)([^\.]+)(?:\.|$)"  # First sentence or fragment
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, response, re.IGNORECASE)
            if matches:
                return matches.group(1).strip()
        
        # If no pattern matches, return the first 100 characters
        return response[:100] + ("..." if len(response) > 100 else "")
