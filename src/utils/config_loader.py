"""Configuration loader for RAG system."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """Configuration manager for RAG system."""

    def __init__(self, config_path: str = None):
        """Initialize configuration.

        Args:
            config_path: Path to settings.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config/settings.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "settings.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

        # Load environment variables
        load_dotenv()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default=None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., "embeddings.model")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_api_key(self, service: str) -> str:
        """Get API key from environment variables.

        Args:
            service: Service name (e.g., "openai", "anthropic")

        Returns:
            API key

        Raises:
            ValueError: If API key not found
        """
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "cohere": "COHERE_API_KEY",
            "voyage": "VOYAGE_API_KEY",
            "qdrant": "QDRANT_API_KEY",
        }

        env_var = env_var_map.get(service.lower())
        if not env_var:
            raise ValueError(f"Unknown service: {service}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"{env_var} not found in environment variables. "
                f"Please set it in your .env file."
            )

        return api_key

    @property
    def embeddings(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return self._config.get("embeddings", {})

    @property
    def qdrant(self) -> Dict[str, Any]:
        """Get Qdrant configuration."""
        return self._config.get("qdrant", {})

    @property
    def chunking(self) -> Dict[str, Any]:
        """Get chunking configuration."""
        return self._config.get("chunking", {})

    @property
    def retrieval(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return self._config.get("retrieval", {})

    @property
    def query_processing(self) -> Dict[str, Any]:
        """Get query processing configuration."""
        return self._config.get("query_processing", {})

    @property
    def llm(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        return self._config.get("llm", {})

    @property
    def evaluation(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self._config.get("evaluation", {})

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata extraction configuration."""
        return self._config.get("metadata", {})

    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance.

    Args:
        config_path: Path to settings.yaml. If None, uses default.

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
