import os
import logging
from typing import Dict, Any
import requests

logger = logging.getLogger(__name__)

class DeepSeekAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        self.api_base_url = "https://api.deepseek.com/v1"
        
        if not self.api_key:
            logger.warning("DeepSeek API key not found in environment variables")

    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze content using DeepSeek API for topic analysis and fact verification
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Request for topic analysis and fact verification
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": """Analyze this news content for:
                        1. Main topics and themes
                        2. Factual claims and their verifiability
                        3. Potential biases or misleading information
                        Provide a structured analysis with confidence scores."""
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                analysis = response.json()
                return self._parse_analysis(analysis)
            else:
                logger.error(f"DeepSeek API error: {response.status_code} - {response.text}")
                return self._get_default_analysis()
                
        except Exception as e:
            logger.error(f"Error in DeepSeek analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _parse_analysis(self, raw_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the raw API response into structured analysis"""
        try:
            content = raw_analysis['choices'][0]['message']['content']
            return {
                'topic_analysis': content,
                'api_response': True,
                'success': True
            }
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing DeepSeek analysis: {str(e)}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when API fails"""
        return {
            'topic_analysis': "Unable to perform deep analysis at this time",
            'api_response': False,
            'success': False
        }
