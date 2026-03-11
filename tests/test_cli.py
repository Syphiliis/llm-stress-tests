import tempfile
import unittest
from unittest.mock import patch

import cli


class CliConfigTests(unittest.TestCase):
    def test_custom_config_is_not_marked_for_cleanup(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml") as handle:
            with patch("cli.get_input", return_value=handle.name):
                path, should_cleanup = cli.create_dynamic_config(
                    gpu_ip="127.0.0.1",
                    test_choice="5",
                    custom_ports=None,
                    duration_override_seconds=None,
                )

        self.assertEqual(path, handle.name)
        self.assertFalse(should_cleanup)


if __name__ == "__main__":
    unittest.main()
