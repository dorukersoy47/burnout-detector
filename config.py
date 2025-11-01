"""
Configuration file for API keys and settings.
Add this file to .gitignore to keep your keys secure!
"""

import os

class Config:
    """Centralized configuration for the burnout detector system"""

    # OpenAI API Key - Set via environment variable or hardcode (NOT recommended for production)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

    # OpenAI Model Selection
    OPENAI_MODEL = "gpt-4o"  # Options: "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"

    # Other settings
    MAX_INVESTIGATION_ITERATIONS = 15
    INTERVENTION_PLANNING_TEMPERATURE = 0.7
    INVESTIGATION_TEMPERATURE = 0.7

    @classmethod
    def is_openai_configured(cls) -> bool:
        """Check if OpenAI API key is properly configured"""
        # Check if key exists, is not empty, and starts with 'sk-'
        return bool(
            cls.OPENAI_API_KEY and
            len(cls.OPENAI_API_KEY) > 20 and
            cls.OPENAI_API_KEY.startswith("sk-")
        )
