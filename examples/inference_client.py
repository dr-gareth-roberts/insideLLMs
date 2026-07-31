"""Run the blessed model-backed inference path without external API calls."""

from __future__ import annotations

import asyncio
import json

from insideLLMs import InferenceClient


async def main() -> None:
    client = InferenceClient.from_model_config(
        {
            "type": "dummy",
            "args": {"canned_response": "Paris"},
        }
    )
    result = await client.generate("What is the capital of France?")
    print(
        json.dumps(
            {
                "answer": result.answer,
                "strategy": result.provenance["strategy"],
                "model": result.provenance["model"],
                "calls": result.spend.calls,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
