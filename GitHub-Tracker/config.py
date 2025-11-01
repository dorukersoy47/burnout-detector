from dotenv import load_dotenv
import os
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class that loads from environment variables"""
    
    @classmethod
    def from_env(cls):
        github_token = os.getenv("GH_TOKEN")
        github_org = os.getenv("GITHUB_ORG")
        github_repos = os.getenv("GITHUB_REPOS", "")
        team_id = os.getenv("TEAM_ID")
        mock = os.getenv("MOCK", "false").lower() == "true"
        port = int(os.getenv("PORT", 8011))
        host = os.getenv("HOST", "127.0.0.1")  # Add this line

        if not github_org:
            raise ValueError("GITHUB_ORG environment variable is required")
        
        return cls(
            github_token=github_token,
            github_org=github_org,
            github_repos=[r.strip() for r in github_repos.split(",") if r.strip()],
            team_id=team_id,
            mock=mock,
            port=port,
            host=host  # Add this parameter
        )
    
    def __init__(self, github_token: Optional[str], github_org: str,
                 github_repos: list, team_id: str, mock: bool, port: int, host: str = "127.0.0.1"):  # Add host parameter
        self.github_token = github_token
        self.github_org = github_org
        self.github_repos = github_repos
        self.team_id = team_id
        self.mock = mock
        self.port = port
        self.host = host  # Add this line
