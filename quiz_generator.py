from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from langchain_groq import ChatGroq

class QuizGenerator:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500
        )
        self.retry_delay = 5  # Seconds between retries
        self.max_retries = 3
        self.model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

    def generate_quiz(self, pdf_text):
        """
        Generate a quiz from PDF text with error handling and retry logic.
        
        Args:
            pdf_text (str): Text extracted from PDF documents.
        
        Returns:
            list: List of structured quiz questions.
        
        Raises:
            RuntimeError: If quiz generation fails after retries.
        """
        try:
            # Split text into manageable chunks
            chunks = self.text_splitter.split_text(pdf_text)
            context = "\n".join(chunks[:3])  # Use first 3 chunks for context

            prompt = f"""Generate 5 MCQ questions from this context:
            {context}
            
            Return the response ONLY as a valid JSON array of objects with this EXACT structure:
            [
              {{
                "question": "question text here",
                "options": ["A) option 1", "B) option 2", "C) option 3", "D) option 4"],
                "answer": "a"
              }}
            ]
            Do not include any markdown formatting like ```json. Just the raw JSON array."""

            # Retry logic for rate limits
            for attempt in range(self.max_retries):
                try:
                    response = self.model.invoke(prompt)
                    return self.parse_quiz(response.content)
                except Exception as e:
                    if "429" in str(e) and attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise

        except Exception as e:
            raise RuntimeError(f"Quiz generation failed: {str(e)}")

    def parse_quiz(self, response_text):
        """
        Parse the model's response into structured quiz questions.
        
        Args:
            response_text (str): Raw response text from the model.
        
        Returns:
            list: List of structured quiz questions.
        """
        import json
        import re
        
        try:
            # Clean up potential markdown formatting from the response
            clean_text = re.sub(r'```json\s*', '', response_text)
            clean_text = re.sub(r'```\s*$', '', clean_text).strip()
            
            questions = json.loads(clean_text)
            
            # Ensure answer is just the lowercase letter
            for q in questions:
                if 'answer' in q and isinstance(q['answer'], str):
                    q['answer'] = q['answer'].lower().strip()[0]
            
            return [q for q in questions if self._is_valid_question(q)][:5]
        except json.JSONDecodeError:
            return []

    def _is_valid_question(self, question):
        """
        Validate the structure of a quiz question.
        
        Args:
            question (dict): Quiz question to validate.
        
        Returns:
            bool: True if the question is valid, False otherwise.
        """
        return (
            len(question.get("options", [])) == 4 and 
            question.get("answer") in ['a', 'b', 'c', 'd'] and 
            bool(question.get("question"))
        )