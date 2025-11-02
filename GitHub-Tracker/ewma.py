"""
Exponential Weighted Moving Average (EWMA) calculation
Provides smoothing for trend analysis with alpha=0.3
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EWMACalculator:
    """
    Computes EWMA for time series metrics
    Formula: EWMA_t = alpha * value_t + (1 - alpha) * EWMA_t-1
    """

    def __init__(self, alpha: float = 0.3):
        """
        Initialize EWMA calculator

        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
                  0.3 is recommended for 24h intervals
        """
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        self.alpha = alpha
        logger.info(f"✅ EWMA Calculator initialized with alpha={alpha}")

    def compute(self, current_value: float, previous_ewma: Optional[float] = None) -> float:
        """
        Compute EWMA for a metric

        Args:
            current_value: The current observation
            previous_ewma: The previous EWMA value (None for first calculation)

        Returns:
            The new EWMA value
        """
        if previous_ewma is None:
            # First value: EWMA = current value
            ewma = current_value
        else:
            # Standard EWMA formula
            ewma = self.alpha * current_value + (1 - self.alpha) * previous_ewma

        return round(ewma, 2)