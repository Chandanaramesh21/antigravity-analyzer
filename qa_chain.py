from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class QASystem:
    def __init__(self, services):
        self.services = services
        self.search = services['search']
        self.embeddings = services['embeddings']
        self.youtube = services.get('youtube')
        
    def get_answer(self, query: str, vector_store) -> Tuple[str, str, List[Dict]]:
        try:
            docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
            yt_links = self._get_youtube_links(query)
            web_links = self._get_web_links(query)
            all_links = self._combine_links(yt_links, web_links)

            answer = self._generate_answer_with_references(query, docs_with_scores, all_links)
            return answer, "combined", all_links

        except Exception as e:
            return f"Error processing query: {str(e)}", "error", []
        
    def _get_youtube_links(self, query: str) -> List[Dict]:
        try:
            if not self.youtube:
                return []
                
            search_response = self.youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=3,
                type='video'
            ).execute()
            
            return [{
                'type': 'youtube',
                'title': item['snippet']['title'],
                'url': f"https://youtu.be/{item['id']['videoId']}",
                'description': item['snippet']['description'][:200] + '...'
            } for item in search_response.get('items', [])]

        except Exception:
            return []
        
    def _get_web_links(self, query: str) -> List[Dict]:
        try:
            if not self.search:
                return []
                
            results = self.search.results(query, 3)
            return [{
                'type': 'web',
                'title': res['title'],
                'url': res['link'],
                'description': res['snippet'][:200] + '...'
            } for res in results]

        except Exception:
            return []

    def generate_summary(self, vector_store, focus: str = None):
        try:
            docs = vector_store.similarity_search(focus or "summary", k=5)

            summary_chain = load_summarize_chain(
                self.services['llm'],
                chain_type="map_reduce"
            )

            return summary_chain.run(docs), []

        except Exception as e:
            return f"Summary generation failed: {str(e)}", []

    def _web_search(self, query: str):
        try:
            results = self.search.results(query, 3)
            docs = [Document(page_content=res['snippet']) for res in results]

            chain = load_qa_chain(
                self.services['llm'],
                prompt=self._web_prompt()
            )

            return chain.run({"input_documents": docs, "question": query})

        except Exception as e:
            return f"Web search failed: {str(e)}"
        
    def _combine_links(self, yt_links, web_links):
        return (yt_links + web_links)[:5]

    def _generate_answer_with_references(self, query, docs, links):
        try:
            context = "\n".join([doc[0].page_content for doc in docs[:2]])

            prompt = f"""
            Answer the question using the context.

            Question: {query}
            Context: {context}
            """

            model = self.services['llm']
            response = model.invoke(prompt)

            return response.content

        except Exception as e:
            return f"Failed to generate answer: {str(e)}"

    def _web_prompt(self):
        return PromptTemplate.from_template("""
        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

    def generate_image_caption(self, image_bytes, query=None):
        try:
            model = self.services['gemini_vision']
            prompt_text = query if query else "Describe the image in detail."

            response = model.generate_content([
                prompt_text,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])

            return response.text

        except Exception as e:
            return f"Caption error: {str(e)}"

    def process_extracted_text(self, text, query=None):
        try:
            if not query:
                return text

            model = self.services['llm']
            response = model.invoke(f"{text}\n\n{query}")

            return response.content

        except Exception as e:
            return f"Text analysis failed: {str(e)}"