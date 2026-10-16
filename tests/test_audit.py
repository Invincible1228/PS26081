"""
Automated Test for 26-Point Quality and Integrity Audit (PS 26081).
Integrates the deep audit suite into the standard unittest discovery runner.
"""
import sys
from pathlib import Path
_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import unittest
from audit import run_deep_audit


class TestDeepCrossPhaseAudit(unittest.TestCase):
    """Executes the complete 26-point integrity audit as an automated test."""

    def test_all_26_audit_checks_pass(self):
        """Validates that all 26 cross-phase integrity and sanity checks pass."""
        success = run_deep_audit()
        self.assertTrue(success, "One or more deep audit checks failed.")


if __name__ == "__main__":
    unittest.main()
