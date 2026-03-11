import unittest

from src.clients.llama_cpp import LlamaCppClient
from src.config.schema import ServerConfig


class LlamaCppClientTests(unittest.TestCase):
    def test_builds_llama_cpp_payload(self) -> None:
        client = LlamaCppClient(ServerConfig(base_url="http://localhost:8080/completion"))

        payload = client._build_payload("hello", 64)

        self.assertEqual(
            payload,
            {
                "prompt": "hello",
                "stream": True,
                "n_predict": 64,
            },
        )

    def test_builds_ollama_payload(self) -> None:
        client = LlamaCppClient(
            ServerConfig(
                base_url="http://localhost:11434/api/generate",
                model_alias="qwen2.5:7b",
            )
        )

        payload = client._build_payload("hello", 64)

        self.assertEqual(payload["model"], "qwen2.5:7b")
        self.assertEqual(payload["prompt"], "hello")
        self.assertEqual(payload["options"]["num_predict"], 64)
        self.assertTrue(payload["stream"])

    def test_ollama_requires_model_alias(self) -> None:
        client = LlamaCppClient(ServerConfig(base_url="http://localhost:11434/api/generate"))

        with self.assertRaises(ValueError):
            client._build_payload("hello", 64)

    def test_decodes_llama_cpp_sse_chunks(self) -> None:
        client = LlamaCppClient(ServerConfig(base_url="http://localhost:8080/completion"))

        decoded = client._decode_stream_chunk(b'data: {"content":"Hello","stop":false}\n')

        self.assertEqual(decoded, {"content": "Hello", "stop": False})

    def test_decodes_ollama_ndjson_chunks(self) -> None:
        client = LlamaCppClient(
            ServerConfig(
                base_url="http://localhost:11434/api/generate",
                model_alias="qwen2.5:7b",
            )
        )

        decoded = client._decode_stream_chunk(b'{"response":"Hi","done":false}\n')

        self.assertEqual(decoded, {"response": "Hi", "done": False})

    def test_ollama_uses_eval_count_when_present(self) -> None:
        client = LlamaCppClient(
            ServerConfig(
                base_url="http://localhost:11434/api/generate",
                model_alias="qwen2.5:7b",
            )
        )

        count = client._update_output_token_count(2, {"done": True, "eval_count": 17})

        self.assertEqual(count, 17)


if __name__ == "__main__":
    unittest.main()
