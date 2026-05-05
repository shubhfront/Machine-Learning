"""Compatibility wrapper for the renamed classifier service.

The web app now keeps backend code in `percep/app/services/classifier.py`.
This file remains so old imports and your active editor tab still work.
"""

from percep.app.services.classifier import *  # noqa: F403
