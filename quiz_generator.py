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
        self.model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

    def generate_quiz(self, pdf_text):
        """
        Generate a quiz from PDF text with error handling and retry logic.
        """
        try:
            # Split text into manageable chunks
            chunks = self.text_splitter.split_text(pdf_text)
            # Use only the first chunk (10k chars) to avoid hitting Groq's free tier token limits
            context = chunks[0] if chunks else pdf_text

            prompt = f"""Generate 5 MCQ questions from this context:
            {context}
            
            Return the response ONLY as a valid JSON array of objects with this EXACT structure:
            [
              {{
                "question": "question text here",
                "options": ["A) option 1", "B) option 2", "C) option 3", "D) option 4"],
                "answer": "a",
                "explanation": "Explanation for why A is correct."
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
        """
        import json
        import re
        
        try:
            # Extract JSON array from anywhere in the text to handle conversational filler
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx == -1 or end_idx == -1:
                return []
                
            json_str = response_text[start_idx:end_idx+1]
            questions = json.loads(json_str)
            
            # Ensure answer is just the lowercase letter
            for q in questions:
                if 'answer' in q and isinstance(q['answer'], str):
                    # Search for the first a, b, c, or d
                    match = re.search(r'[a-d]', q['answer'].lower())
                    if match:
                        q['answer'] = match.group(0)
                    else:
                        q['answer'] = 'a' # Fallback
            
            return [q for q in questions if self._is_valid_question(q)][:5]
        except Exception as e:
            print(f"Quiz parsing error: {e}")
            return []

    def _is_valid_question(self, question):
        """
        Validate the structure of a quiz question.
        """
        return (
            len(question.get("options", [])) >= 2 and 
            bool(question.get("answer")) and 
            bool(question.get("question"))
        )