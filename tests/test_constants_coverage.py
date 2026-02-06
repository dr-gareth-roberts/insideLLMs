"""Tests for insideLLMs.constants module to achieve coverage."""

from insideLLMs.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    LOGGER_PREFIX,
    MAX_RETRY_DELAY,
    MAX_TIMEOUT,
    AnthropicModels,
    ArtifactNames,
    CacheDefaults,
    CohereModels,
    EnvVars,
    FileExtensions,
    GeminiModels,
    OpenAIModels,
    RateLimitDefaults,
)


class TestTimeoutConstants:
    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 60.0

    def test_default_max_retries(self):
        assert DEFAULT_MAX_RETRIES == 2

    def test_max_timeout(self):
        assert MAX_TIMEOUT == 600.0

    def test_default_retry_delay(self):
        assert DEFAULT_RETRY_DELAY == 1.0

    def test_max_retry_delay(self):
        assert MAX_RETRY_DELAY == 60.0


class TestTokenConstants:
    def test_default_max_tokens(self):
        assert DEFAULT_MAX_TOKENS == 1024

    def test_default_temperature(self):
        assert DEFAULT_TEMPERATURE == 0.7


class TestOpenAIModels:
    def test_gpt4_turbo(self):
        assert OpenAIModels.GPT_4_TURBO == "gpt-4-turbo"

    def test_gpt4(self):
        assert OpenAIModels.GPT_4 == "gpt-4"

    def test_gpt4o(self):
        assert OpenAIModels.GPT_4O == "gpt-4o"

    def test_gpt4o_mini(self):
        assert OpenAIModels.GPT_4O_MINI == "gpt-4o-mini"

    def test_gpt35_turbo(self):
        assert OpenAIModels.GPT_35_TURBO == "gpt-3.5-turbo"

    def test_default(self):
        assert OpenAIModels.DEFAULT == OpenAIModels.GPT_35_TURBO


class TestAnthropicModels:
    def test_claude3_opus(self):
        assert AnthropicModels.CLAUDE_3_OPUS == "claude-3-opus-20240229"

    def test_claude35_sonnet(self):
        assert AnthropicModels.CLAUDE_35_SONNET == "claude-3-5-sonnet-20240620"

    def test_claude3_sonnet(self):
        assert AnthropicModels.CLAUDE_3_SONNET == "claude-3-sonnet-20240229"

    def test_claude3_haiku(self):
        assert AnthropicModels.CLAUDE_3_HAIKU == "claude-3-haiku-20240307"

    def test_default(self):
        assert AnthropicModels.DEFAULT == AnthropicModels.CLAUDE_3_OPUS


class TestGeminiModels:
    def test_gemini_pro(self):
        assert GeminiModels.GEMINI_PRO == "gemini-pro"

    def test_gemini_15_pro(self):
        assert GeminiModels.GEMINI_15_PRO == "gemini-1.5-pro"

    def test_gemini_15_flash(self):
        assert GeminiModels.GEMINI_15_FLASH == "gemini-1.5-flash"

    def test_default(self):
        assert GeminiModels.DEFAULT == GeminiModels.GEMINI_PRO


class TestCohereModels:
    def test_command(self):
        assert CohereModels.COMMAND == "command"

    def test_command_light(self):
        assert CohereModels.COMMAND_LIGHT == "command-light"

    def test_command_r(self):
        assert CohereModels.COMMAND_R == "command-r"

    def test_command_r_plus(self):
        assert CohereModels.COMMAND_R_PLUS == "command-r-plus"

    def test_default(self):
        assert CohereModels.DEFAULT == CohereModels.COMMAND


class TestEnvVars:
    def test_openai_key(self):
        assert EnvVars.OPENAI_API_KEY == "OPENAI_API_KEY"

    def test_anthropic_key(self):
        assert EnvVars.ANTHROPIC_API_KEY == "ANTHROPIC_API_KEY"

    def test_google_key(self):
        assert EnvVars.GOOGLE_API_KEY == "GOOGLE_API_KEY"

    def test_cohere_key(self):
        assert EnvVars.COHERE_API_KEY == "COHERE_API_KEY"

    def test_hf_token(self):
        assert EnvVars.HF_TOKEN == "HF_TOKEN"

    def test_run_root(self):
        assert EnvVars.INSIDELLMS_RUN_ROOT == "INSIDELLMS_RUN_ROOT"

    def test_disable_plugins(self):
        assert EnvVars.INSIDELLMS_DISABLE_PLUGINS == "INSIDELLMS_DISABLE_PLUGINS"


class TestCacheDefaults:
    def test_max_size(self):
        assert CacheDefaults.MAX_SIZE == 1000

    def test_ttl_seconds(self):
        assert CacheDefaults.TTL_SECONDS == 3600

    def test_disk_max_size_mb(self):
        assert CacheDefaults.DISK_MAX_SIZE_MB == 100

    def test_hash_algorithm(self):
        assert CacheDefaults.HASH_ALGORITHM == "sha256"


class TestRateLimitDefaults:
    def test_requests_per_minute(self):
        assert RateLimitDefaults.REQUESTS_PER_MINUTE == 60

    def test_tokens_per_minute(self):
        assert RateLimitDefaults.TOKENS_PER_MINUTE == 90000

    def test_burst_size(self):
        assert RateLimitDefaults.BURST_SIZE == 10


class TestLoggerPrefix:
    def test_prefix(self):
        assert LOGGER_PREFIX == "insideLLMs"


class TestFileExtensions:
    def test_jsonl(self):
        assert FileExtensions.JSONL == ".jsonl"

    def test_json(self):
        assert FileExtensions.JSON == ".json"

    def test_yaml(self):
        assert FileExtensions.YAML == ".yaml"

    def test_yml(self):
        assert FileExtensions.YML == ".yml"

    def test_html(self):
        assert FileExtensions.HTML == ".html"

    def test_csv(self):
        assert FileExtensions.CSV == ".csv"


class TestArtifactNames:
    def test_records(self):
        assert ArtifactNames.RECORDS == "records.jsonl"

    def test_manifest(self):
        assert ArtifactNames.MANIFEST == "manifest.json"

    def test_config_resolved(self):
        assert ArtifactNames.CONFIG_RESOLVED == "config.resolved.yaml"

    def test_summary(self):
        assert ArtifactNames.SUMMARY == "summary.json"

    def test_report(self):
        assert ArtifactNames.REPORT == "report.html"

    def test_diff(self):
        assert ArtifactNames.DIFF == "diff.json"

    def test_run_marker(self):
        assert ArtifactNames.RUN_MARKER == ".insidellms_run"
