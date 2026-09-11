# Provider declarations and offline diagnostics

`insideLLMs.models.catalogue.PROVIDER_CATALOGUE` is the immutable source for
builtin model registration and doctor prerequisites. Its frozen `ProviderSpec`,
`DependencySpec` and `ProviderCapabilities` values contain only static data.
Reading the catalogue never imports a provider SDK or constructs a client.
Metadata is kept separate from registry constructor defaults.

Run discovery without importing third-party plugin registration code at package
startup:

```bash
INSIDELLMS_DISABLE_PLUGINS=1 insidellms doctor --capabilities --format json
```

Doctor itself discovers entry-point names without loading them. Existing plugin
registrations can be inspected; normal package startup still supports automatic
plugin loading when the environment switch is absent. The switch is necessary
when installed third-party startup code must not execute.

## Builtin prerequisites

Module discovery does not prove that an SDK imports successfully, a credential is
accepted, an endpoint is reachable, or a model supports the selected operation.
Credentials below describe the adapter's default environment lookup; constructor
credentials, custom `api_key_env` values and endpoint overrides are not evaluated.

| Registry key | Required modules (distribution) | Credential environment alternatives | External requirements, not checked |
| --- | --- | --- | --- |
| `dummy` | None | None | None |
| `openai` | `openai` (`openai`) | `OPENAI_API_KEY` | Configured OpenAI-compatible endpoint |
| `openrouter` | `openai` (`openai`) | `OPENROUTER_API_KEY` | OpenRouter endpoint |
| `anthropic` | `anthropic` (`anthropic`) | `ANTHROPIC_API_KEY` | Anthropic endpoint |
| `gemini` | `google.generativeai` (`google-generativeai`) | `GOOGLE_API_KEY` | Gemini endpoint |
| `cohere` | `cohere` (`cohere`) | `CO_API_KEY` or `COHERE_API_KEY` | Cohere endpoint |
| `huggingface` | `transformers` (`transformers`), `torch` (`torch`) | None | Compatible weights, tokenizer and compute resources |
| `llamacpp` | `llama_cpp` (`llama-cpp-python`) | None | Compatible GGUF file and compute resources |
| `ollama` | `ollama` (`ollama`) | None; `OLLAMA_API_KEY` is optional | Ollama service with requested model |
| `vllm` | `openai` (`openai`) | None by default | vLLM endpoint with requested model |

Cohere accepts either credential independently. Each nested group in
`credential_alternatives` means “at least one of these variables”; all groups
must be satisfied. Unsatisfied groups appear as readable `or` expressions in
`missing_credentials`. Only variable names and presence-derived booleans are
reported, never credential values.

## Declared operations

The operation labels describe this adapter's implementation. They do not certify
every remote model or endpoint. Runtime `can_chat`, `can_stream` and async
dispatch predicates are unchanged and continue to inspect the actual instance.

| Adapter | Generate | Chat | Stream |
| --- | --- | --- | --- |
| `dummy` | Native local implementation | Native local implementation | Simulated by splitting the generated response |
| `huggingface` | Native | Simulated by concatenating messages | Simulated as one completed response |
| Other catalogued adapters | Native | Native | Native |

All current builtin adapters inherit simulated `batch_generate` behavior. Their
native `agenerate`, `achat` and `astream` operations are unsupported. Wrappers or
pipelines may expose additional callable operations; this table describes the
base adapters. Dummy responses are artificial test data regardless of operation
labels.

## Interpreting doctor JSON

The existing `checks`, `warnings`, capability collections and default exit policy
remain available. Provider entries add:

- `metadata_status`: `declared` for a canonical builtin registration, `unknown`
  for an unrecognised registration or a plugin replacing a builtin factory.
- `dependencies`: module/distribution pairs and module-discovery availability.
- `credential_alternatives`, `optional_credentials`, `external_requirements`:
  declarations from the catalogue.
- `declared_capabilities`: native, simulated or unsupported operations.
- `prerequisites_ready`: discovered SDK modules and present default credentials.
- `live_verification`: always `not_checked`; doctor makes no provider calls.
- `budget_support`: `unknown`; an installed adapter or declared operation is not
  evidence of enforceable prices, request bounds or billing limits. A separate
  invocation budget policy must make that decision.

`status: ready` means only that the local prerequisite checks passed. It does not
assert that external requirements were checked. Unknown plugin entries use
`status: unknown`, null prerequisite/capability fields and `live_verification:
not_checked`; they are never classified as ready from missing metadata.

Builtin provenance compares the registered factory object with the canonical
factory captured during registration. A familiar name alone is insufficient.
`get_builtin_model_factory` and `get_builtin_probe_factory` expose these objects
for identity comparisons without importing or invoking adapters. Strict callers
must also validate registration defaults and supplied constructor arguments.

By default doctor exits zero even when optional prerequisites are missing.
`--fail-on-warn` retains its existing handling of the top-level diagnostic
warnings; provider capability entries do not silently become new exit gates.
