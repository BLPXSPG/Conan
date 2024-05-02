"""Configuration class to store the state of bools for different scripts access."""
import os
from typing import List

import openai
import yaml
from colorama import Fore

from config.singleton import Singleton
from dotenv import load_dotenv
# Load default environment variables (.env)
load_dotenv()

GPT_4_MODEL = "gpt-4"
GPT_3_MODEL = "gpt-3.5-turbo"

class Config(metaclass=Singleton):
    """
    Configuration class to store the state of bools for different scripts access.
    """

    def __init__(self) -> None:
        """Initialize the Config class"""
        self.workspace_path: str = None
        self.file_logger_path: str = None
        self.max_tokens = 4096-1000
        # self.max_tokens = 128000
        self.model_llama_13b_path='/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496'
        self.model_llama_70b_path="/scratch/prj/lmrep/llama2_model/Llama-2-70b-chat-hf"
        self.model_llama_3_70b_path="/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/0cac6d727e4cdf117e1bde11e4c7badd8b963919"

        # self.model="llama-chat-70b"
        self.model="llama-chat-13b"
        # self.model="gpt-3.5-turbo"
        # self.model = "gpt-4"

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False
        self.skip_reprompt = False
        self.allow_downloads = False
        self.skip_news = False
        
        self.authorise_key = os.getenv("AUTHORISE_COMMAND_KEY", "y")
        self.exit_key = os.getenv("EXIT_KEY", "n")
        self.plain_output = os.getenv("PLAIN_OUTPUT", "False") == "True"

        disabled_command_categories = os.getenv("DISABLED_COMMAND_CATEGORIES")
        if disabled_command_categories:
            self.disabled_command_categories = disabled_command_categories.split(",")
        else:
            self.disabled_command_categories = []

        deny_commands = os.getenv("DENY_COMMANDS")
        if deny_commands:
            self.deny_commands = deny_commands.split(",")
        else:
            self.deny_commands = []

        allow_commands = os.getenv("ALLOW_COMMANDS")
        if allow_commands:
            self.allow_commands = allow_commands.split(",")
        else:
            self.allow_commands = []

        self.ai_settings_file = os.getenv("AI_SETTINGS_FILE", "ai_settings.yaml")
        self.prompt_settings_file = os.getenv(
            "PROMPT_SETTINGS_FILE", "prompt_settings.yaml"
        )
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-4")
        self.fast_token_limit = int(os.getenv("FAST_TOKEN_LIMIT", 4000))
        self.smart_token_limit = int(os.getenv("SMART_TOKEN_LIMIT", 8000))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.browse_spacy_language_model = os.getenv(
            "BROWSE_SPACY_LANGUAGE_MODEL", "zh_core_web_sm"
        )

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.execute_local_commands = (
            os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True"
        )
        self.restrict_to_workspace = (
            os.getenv("RESTRICT_TO_WORKSPACE", "True") == "True"
        )
        if self.use_azure:
            self.load_azure_config()
            openai.api_type = self.openai_api_type
            openai.api_base = self.openai_api_base
            openai.api_version = self.openai_api_version
            openai.api_key = self.azure_api_key

        if self.openai_organization is not None:
            openai.organization = self.openai_organization
        
        self.picture_generation_backend = "model"
        self.model_id = "prompthero/openjourney"
        self.hotpot_api_key = ""

        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_voice_1_id = os.getenv("ELEVENLABS_VOICE_1_ID")
        self.elevenlabs_voice_2_id = os.getenv("ELEVENLABS_VOICE_2_ID")

        self.use_mac_os_tts = False
        self.use_mac_os_tts = os.getenv("USE_MAC_OS_TTS")

        self.chat_messages_enabled = os.getenv("CHAT_MESSAGES_ENABLED") == "True"

        self.use_brian_tts = False
        self.use_brian_tts = os.getenv("USE_BRIAN_TTS")

        self.github_api_key = os.getenv("GITHUB_API_KEY")
        self.github_username = os.getenv("GITHUB_USERNAME")

        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.custom_search_engine_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.image_size = int(os.getenv("IMAGE_SIZE", 256))
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.huggingface_image_model = os.getenv(
            "HUGGINGFACE_IMAGE_MODEL", "CompVis/stable-diffusion-v1-4"
        )
        self.huggingface_audio_to_text_model = os.getenv(
            "HUGGINGFACE_AUDIO_TO_TEXT_MODEL"
        )
        self.sd_webui_url = os.getenv("SD_WEBUI_URL", "http://localhost:7860")
        self.sd_webui_auth = os.getenv("SD_WEBUI_AUTH")

        # Selenium browser settings
        self.selenium_web_browser = os.getenv("USE_WEB_BROWSER", "chrome")
        self.selenium_headless = os.getenv("HEADLESS_BROWSER", "True") == "True"

        # User agent header to use when making HTTP requests
        # Some websites might just completely deny request with an error code if
        # no user agent was found.
        self.user_agent = os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        )

        self.memory_backend = os.getenv("MEMORY_BACKEND", "json_file")
        self.memory_index = os.getenv("MEMORY_INDEX", "auto-gpt-memory")

        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == "True"

    def get_openai_credentials(self, model: str) -> dict[str, str]:
        credentials = {
            "api_key": self.openai_api_key,
            "organization": self.openai_organization,
        }
        if self.use_azure:
            azure_credentials = self.get_azure_credentials(model)
            credentials.update(azure_credentials)
        return credentials
    
    def get_azure_credentials(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""
        #deployment_id = self.get_azure_deployment_id_for_model(model)

        self.load_azure_config()
        kwargs = {
            "api_key": self.azure_api_key,
            "api_type": self.openai_api_type,
            "api_base": self.openai_api_base,
            "api_version": self.openai_api_version,
            "engine": self.get_azure_deployment_id_for_model(model)
        }
        #if model == self.embedding_model:
        #    kwargs["engine"] = deployment_id
        #else:
        #    kwargs["deployment_id"] = deployment_id
        return kwargs


    def get_azure_deployment_id_for_model(self, model: str) -> str:
        """
        Returns the relevant deployment id for the model specified.

        Parameters:
            model(str): The model to map to the deployment id.

        Returns:
            The matching deployment id if found, otherwise an empty string.
        """
        if model == self.fast_llm_model:
            return self.azure_model_to_deployment_id_map[
                "fast_llm_model_deployment_id"
            ]  # type: ignore
        elif model == self.smart_llm_model:
            return self.azure_model_to_deployment_id_map[
                "smart_llm_model_deployment_id"
            ]  # type: ignore
        elif model == "text-embedding-ada-002":
            return self.azure_model_to_deployment_id_map[
                "embedding_model_deployment_id"
            ]  # type: ignore
        else:
            return ""

    AZURE_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "azure.yaml")

    def load_azure_config(self, config_file: str = AZURE_CONFIG_FILE) -> None:
        """
        Loads the configuration parameters for Azure hosting from the specified file
          path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file. DEFAULT: "../azure.yaml"

        Returns:
            None
        """
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        self.openai_api_type = config_params.get("azure_api_type") or "azure"
        self.openai_api_base = config_params.get("azure_api_base") or ""
        self.openai_api_version = (
            config_params.get("azure_api_version") or "2023-03-15-preview"
        )
        self.azure_api_key = config_params.get("azure_api_key") 
        self.azure_model_to_deployment_id_map = config_params.get("azure_model_map", {})

    def set_continuous_mode(self, value: bool) -> None:
        """Set the continuous mode value."""
        self.continuous_mode = value

    def set_continuous_limit(self, value: int) -> None:
        """Set the continuous limit value."""
        self.continuous_limit = value

    def set_speak_mode(self, value: bool) -> None:
        """Set the speak mode value."""
        self.speak_mode = value

    def set_fast_llm_model(self, value: str) -> None:
        """Set the fast LLM model value."""
        self.fast_llm_model = value

    def set_smart_llm_model(self, value: str) -> None:
        """Set the smart LLM model value."""
        self.smart_llm_model = value

    def set_fast_token_limit(self, value: int) -> None:
        """Set the fast token limit value."""
        self.fast_token_limit = value

    def set_smart_token_limit(self, value: int) -> None:
        """Set the smart token limit value."""
        self.smart_token_limit = value

    def set_embedding_model(self, value: str) -> None:
        """Set the model to use for creating embeddings."""
        self.embedding_model = value

    def set_openai_api_key(self, value: str) -> None:
        """Set the OpenAI API key value."""
        self.openai_api_key = value

    def set_elevenlabs_api_key(self, value: str) -> None:
        """Set the ElevenLabs API key value."""
        self.elevenlabs_api_key = value

    def set_elevenlabs_voice_1_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 1 ID value."""
        self.elevenlabs_voice_1_id = value

    def set_elevenlabs_voice_2_id(self, value: str) -> None:
        """Set the ElevenLabs Voice 2 ID value."""
        self.elevenlabs_voice_2_id = value

    def set_google_api_key(self, value: str) -> None:
        """Set the Google API key value."""
        self.google_api_key = value

    def set_custom_search_engine_id(self, value: str) -> None:
        """Set the custom search engine id value."""
        self.custom_search_engine_id = value

    def set_debug_mode(self, value: bool) -> None:
        """Set the debug mode value."""
        self.debug_mode = value

    def set_temperature(self, value: int) -> None:
        """Set the temperature value."""
        self.temperature = value

    def set_memory_backend(self, name: str) -> None:
        """Set the memory backend name."""
        self.memory_backend = name
    
    def set_picture_generation_backend(self, value: str) -> None:
        """Set the memory backend name."""
        self.picture_generation_backend = value
    
    def set_picture_generation_model_id(self, value: str) -> None:
        self.model_id = value
    
    def set_hotpot_api_key(self, value:str) -> None:
        self.hotpot_api_key = value

def check_openai_api_key() -> None:
    """Check if the OpenAI API key is set in config.py or as an environment variable."""
    cfg = Config()
    if not cfg.openai_api_key:
        print(
            Fore.RED
            + "Please set your OpenAI API key in .env or as an environment variable."
            + Fore.RESET
        )
        print("You can get your key from https://platform.openai.com/account/api-keys")
        exit(1)