import os
from langchain.utilities import GoogleSearchAPIWrapper
from googleapiclient.discovery import build
from google.cloud import vision
from google.oauth2 import service_account
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import google.generativeai as genai

def initialize_services():
    """Initialize all services with proper credentials"""
    services = {}
    
    try:
        # Validate environment variables
        google_api_key = os.getenv("GOOGLE_API_KEY")
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not all([google_api_key, youtube_api_key, google_cse_id, credentials_path]):
            raise ValueError("Missing required Google API keys or credentials in environment variables")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment variables")

        # Initialize Vision API with service account
        vision_credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-vision']
        )
        services['vision'] = vision.ImageAnnotatorClient(credentials=vision_credentials)

        # Initialize YouTube API service
        services['youtube'] = build('youtube', 'v3', developerKey=youtube_api_key)

        # Initialize Google Search
        services['search'] = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
            k=5
        )

        # Initialize other services (Groq, etc.)
        services['llm'] = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)
        services['embeddings'] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Gemini specifically for Vision (since Groq decommissioned theirs)
        genai.configure(api_key=google_api_key)
        services['gemini_vision'] = genai.GenerativeModel('gemini-2.5-flash')
        
        return services
        
    except Exception as e:
        raise RuntimeError(f"Service initialization failed: {str(e)}")

















# # config.py
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, HarmBlockThreshold, HarmCategory
# from langchain.utilities import GoogleSearchAPIWrapper
# from google.cloud import vision
# from google.oauth2 import service_account
# import google.generativeai as genai

# def initialize_services():
#     """Initialize all Google services with proper credentials"""
#     services = {}
    
#     try:
#         # Load and validate environment variables
#         google_api_key = os.getenv("GOOGLE_API_KEY")
#         if not google_api_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables. Check your .env file.")
        
#         google_cse_id = os.getenv("GOOGLE_CSE_ID")
#         if not google_cse_id:
#             raise ValueError("GOOGLE_CSE_ID not found in environment variables. Required for Google Search.")
        
#         credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
#         if not credentials_path:
#             raise FileNotFoundError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set.")
#         if not os.path.exists(credentials_path):
#             raise FileNotFoundError(
#                 f"Service account JSON file not found at {credentials_path}. Verify the path in .env."
#             )

#         # Configure Vision API with service account
#         vision_credentials = service_account.Credentials.from_service_account_file(
#             credentials_path,
#             scopes=['https://www.googleapis.com/auth/cloud-vision']
#         )
#         vision_client = vision.ImageAnnotatorClient(credentials=vision_credentials)

#         # Configure Gemini (uses API key, not service account)
#         genai.configure(api_key=google_api_key)

#         # Initialize Google Search API wrapper
#         search = GoogleSearchAPIWrapper(
#             google_api_key=google_api_key,
#             google_cse_id=google_cse_id
#         )

#         # Initialize Gemini embeddings with safety settings disabled for specific category
#         embeddings = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=google_api_key,
#             safety_settings={
#                 HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
#             }
#         )

#         # Initialize Gemini Pro model
#         gemini = genai.GenerativeModel('gemini-pro')

#         services.update({
#             'search': search,
#             'embeddings': embeddings,
#             'gemini': genai.GenerativeModel('gemini-1.5-flash'),
#             'vision': vision_client
#         })
        
#         return services
        
#     except (ValueError, FileNotFoundError, ImportError) as e:
#         # Re-raise known exceptions with specific messages
#         raise RuntimeError(f"Configuration error: {str(e)}")
#     except Exception as e:
#         # Catch-all for unexpected errors
#         raise RuntimeError(f"Unexpected error during service initialization: {str(e)}")

























































