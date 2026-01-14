import ollama
from typing import Dict, List, Optional
from duckduckgo_search import DDGS
from datetime import datetime

class ChatManager:
    """
    ChatManager handles interaction with LLMs (via Ollama),
    optional web search context, and maintains chat history.
    """
    
    def __init__(self):
        self.available_models = ["llama3.1:8b", "mistral:7b", "gemma:7b", "phi3:mini"]
        self.current_model = "llama3.1:8b"
        self.chat_history: List[Dict] = []

    def generate_response(
        self,
        question: str,
        context: str = "",
        use_web_search: bool = False
    ) -> Dict:
        """
        Generate a response using the selected LLM and optional web search context.
        """
        # Optional web search context
        web_section = ""
        if use_web_search:
            web_section = self._web_search(question)
            if web_section:
                web_section = f"Recent web search results:\n{web_section}"

        # Prepare prompt
        prompt = f"""You are a helpful research assistant. Use the following context to answer the question.

Context from documents:
{context}

{web_section}

Question: {question}

Answer:"""

        # Call Ollama
        try:
            response = ollama.chat(
                model=self.current_model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            # Safely access content
            answer_text = response.get('message', {}).get('content', '')

        except Exception as e:
            answer_text = f"[Error generating response: {e}]"
            print(f"Ollama chat error: {e}")

        # Store chat history
        self.chat_history.append({
            "question": question,
            "answer": answer_text,
            "model": self.current_model,
            "timestamp": datetime.utcnow().isoformat()
        })

        # Return response with simple confidence heuristic
        return {
            "answer": answer_text,
            "confidence": self._calculate_confidence(answer_text),
            "model": self.current_model
        }

    def _web_search(self, query: str) -> str:
        """
        Perform a quick web search using DuckDuckGo.
        Returns top 3 results as formatted string.
        """
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=3):
                    results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}")
                return "\n\n".join(results)
        except Exception as e:
            print(f"Web search failed: {e}")
            return ""

    def _calculate_confidence(self, answer: str) -> float:
        """
        Simple heuristic for confidence scoring based on answer content.
        """
        indicators = [
            "I don't know",
            "I'm not sure",
            "cannot answer",
            "based on the information"
        ]

        confidence = 0.8  # Base confidence
        if any(indicator.lower() in answer.lower() for indicator in indicators):
            confidence -= 0.3
        if len(answer.split()) > 50:
            confidence += 0.1

        return min(max(confidence, 0.1), 0.99)

    def switch_model(self, model_name: str) -> bool:
        """
        Switch between available LLMs.
        Returns True if successful, False if model not found.
        """
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        return False

    def get_chat_history(self) -> List[Dict]:
        """
        Return the last 10 chat interactions.
        """
        return self.chat_history[-10:]
