"""
conftest.py — pytest root configuration.

This file marks the project root for pytest's Python path resolution.
Having conftest.py at the root allows pytest to correctly import
modules from the 'app' package without requiring PYTHONPATH manipulation
or editable installs.

Shared fixtures for tests are defined in tests/test_api_integration.py.
This file is intentionally kept minimal.
"""