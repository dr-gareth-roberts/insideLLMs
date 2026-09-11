"""Dependency-free two-sided Student-t tails via regularized incomplete beta.

Definitions, symmetry and fraction: https://dlmf.nist.gov/8.17 (2--4, 22--23).
Distribution semantics: https://www.itl.nist.gov/div898/handbook/eda/section3/eda3664.htm
"""

import math

_TOLERANCE = 1e-14
_MAX_ITERATIONS = 10_000
_MIN_DENOMINATOR = 1e-300


def _guard_denominator(value: float) -> float:
    return math.copysign(_MIN_DENOMINATOR, value) if abs(value) < _MIN_DENOMINATOR else value


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate DLMF 8.17.22--23 using modified Lentz iteration."""
    c = 1.0
    d = 1.0 / _guard_denominator(1.0 - (a + b) * x / (a + 1.0))
    fraction = d
    for m in range(1, _MAX_ITERATIONS + 1):
        twice_m = 2 * m
        even = m * (b - m) * x / ((a + twice_m - 1) * (a + twice_m))
        odd = -(a + m) * (a + b + m) * x / ((a + twice_m) * (a + twice_m + 1))
        for coefficient in (even, odd):
            d = 1.0 / _guard_denominator(1.0 + coefficient * d)
            c = _guard_denominator(1.0 + coefficient / c)
            change = d * c
            fraction *= change
        if abs(change - 1.0) <= _TOLERANCE:
            if math.isfinite(fraction) and fraction > 0:
                return fraction
            raise ArithmeticError("Invalid incomplete-beta continued fraction")
    raise ArithmeticError("Incomplete-beta continued fraction did not converge")


def _checked_probability(probability: float) -> float:
    if not math.isfinite(probability) or not -_TOLERANCE <= probability <= 1 + _TOLERANCE:
        raise ArithmeticError("Student-t probability outside [0, 1]")
    return min(1.0, max(0.0, probability))


def two_sided_t_probability(statistic: float, degrees_of_freedom: float) -> float:
    """Return P(|T_df| >= |statistic|) for finite t and finite positive df.

    Preserve both log arguments even when x or 1-x rounds to an endpoint:
    for example x underflows at df=1, t=1e200 but its tail is representable.
    Invalid parameters raise ValueError; numerical failure raises ArithmeticError.
    """
    if not math.isfinite(statistic):
        raise ValueError("Student-t statistic must be finite")
    if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
        raise ValueError("Student-t degrees of freedom must be finite and positive")
    if statistic == 0:
        return 1.0

    # Square only a ratio <= 1, retaining its logarithm through underflow.
    log_ratio = math.log(abs(statistic)) - 0.5 * math.log(degrees_of_freedom)
    squared_ratio = math.exp(-2 * abs(log_ratio))
    log_scale = math.log1p(squared_ratio)
    small = squared_ratio / (1 + squared_ratio)
    if log_ratio >= 0:
        x, complement = small, 1 / (1 + squared_ratio)
        log_x, log_complement = -2 * log_ratio - log_scale, -log_scale
    else:
        x, complement = 1 / (1 + squared_ratio), small
        log_x, log_complement = -log_scale, 2 * log_ratio - log_scale

    a, b = degrees_of_freedom / 2, 0.5
    use_complement = x >= (a + 1) / (a + b + 2)
    if use_complement:
        a, b, x = b, a, complement
        log_x, log_complement = log_complement, log_x
    # log1p(-x) avoids cancellation, but the separately retained complement
    # remains necessary when either argument has rounded to an endpoint.
    if x < 1:
        log_complement = math.log1p(-x)
    log_front = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * log_x
        + b * log_complement
        - math.log(a)
    )
    fraction = _beta_continued_fraction(a, b, x)
    log_tail = log_front + math.log(fraction)
    probability = -math.expm1(log_tail) if use_complement else math.exp(log_tail)
    return _checked_probability(probability)
