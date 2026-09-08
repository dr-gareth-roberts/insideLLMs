"""Independent Student-t references; no optional numerical dependency needed."""

import math

import pytest

from insideLLMs.analysis.statistics import paired_t_test, welchs_t_test

# SciPy 1.18.0: 2 * scipy.stats.t.sf(abs(t), df), independently generated.
# NIST DLMF https://dlmf.nist.gov/8.17 supplies the beta identity.
REFERENCE_GRID = [
    (1, 0.0, 1.0),
    (1, 0.1, 0.93654896513889296),
    (1, 1.0, 0.50000000000000011),
    (1, 2.0, 0.29516723530086653),
    (1, 5.0, 0.1256659163780024),
    (1, 20.0, 0.031804502512352749),
    (2, 0.0, 1.0),
    (2, 0.1, 0.9294654384141402),
    (2, 1.0, 0.42264973081037421),
    (2, 2.0, 0.18350341907227394),
    (2, 5.0, 0.037749551350623724),
    (2, 20.0, 0.002490663892367097),
    (5, 0.0, 1.0),
    (5, 0.1, 0.92423014115466051),
    (5, 1.0, 0.36321746764912272),
    (5, 2.0, 0.10193947882985832),
    (5, 5.0, 0.0041047159800533216),
    (5, 20.0, 5.7755163732241715e-06),
    (30, 0.0, 1.0),
    (30, 0.1, 0.92100961179027119),
    (30, 1.0, 0.32530861542603001),
    (30, 2.0, 0.054625044962983094),
    (30, 5.0, 2.3296685467007803e-05),
    (30, 20.0, 6.7490836657712914e-19),
    (1000, 0.0, 1.0),
    (1000, 0.1, 0.92036436902360408),
    (1000, 1.0, 0.31755241808467227),
    (1000, 2.0, 0.045770346493251617),
    (1000, 5.0, 6.7672563646486285e-07),
    (1000, 20.0, 4.0622884995248135e-75),
]


@pytest.mark.parametrize("df, statistic, expected", REFERENCE_GRID)
def test_two_sided_tail_matches_independent_reference(df, statistic, expected):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    probability = two_sided_t_probability(statistic, df)
    assert probability == pytest.approx(expected, abs=1e-12, rel=1e-8)
    assert probability > 0
    assert probability == pytest.approx(expected, abs=0, rel=1e-8)
    assert two_sided_t_probability(-statistic, df) == probability


@pytest.mark.parametrize("df", [1, 2, 5, 30, 1000])
def test_tail_decreases_with_absolute_statistic(df):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    probabilities = [two_sided_t_probability(t, df) for t in [0, 0.1, 1, 2, 5, 20]]
    assert probabilities[0] == 1.0
    assert all(a > b for a, b in zip(probabilities, probabilities[1:]))


@pytest.mark.parametrize("statistic", [1e-10, 0.1, 1, 2, 20, 1e154, 1e200, 1e308])
def test_cauchy_tail_including_underflowed_beta_argument(statistic):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    expected = (2 / math.pi) * math.atan(1 / statistic)
    probability = two_sided_t_probability(statistic, 1)
    assert probability > 0
    assert probability == pytest.approx(expected, abs=0, rel=1e-8)
    if statistic == 1e-10:
        assert probability < 1.0
        assert probability == pytest.approx(expected, abs=1e-15, rel=0)


@pytest.mark.parametrize("statistic", [0.1, 1, 2, 20, 1e100, 1e154])
def test_df_two_closed_form_retains_small_tails(statistic):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    ratio = math.sqrt(2) / statistic
    root = math.sqrt(1 + ratio * ratio)
    expected = ratio * ratio / (root * (root + 1))
    probability = two_sided_t_probability(statistic, 2)
    assert probability > 0
    assert probability == pytest.approx(expected, abs=0, rel=1e-8)


# mpmath 1.3.0 at 120 digits, independently normalized against df=1/2
# closed forms: betainc(df/2, 1/2, 0, df/(df+t*t), regularized=True).
@pytest.mark.parametrize(
    "df, statistic, expected",
    [
        (1.25, 1e155, 1.2115457207544034e-194),
        (1.25, 1e200, 6.813022261377278e-251),
        (1.5, 1e155, 2.3848964811113156e-233),
        (1.5, 1e200, 7.541704864032493e-301),
    ],
)
def test_noninteger_extreme_tail_keeps_log_argument(df, statistic, expected):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    probability = two_sided_t_probability(statistic, df)
    assert probability > 0
    assert probability == pytest.approx(expected, abs=0, rel=1e-8)


def test_unrepresentable_tail_underflows_to_zero():
    from insideLLMs.analysis._student_t import two_sided_t_probability

    assert two_sided_t_probability(1e200, 2) == 0.0


@pytest.mark.parametrize(
    "statistic, df",
    [
        (math.nan, 1),
        (math.inf, 1),
        (-math.inf, 1),
        (1, math.nan),
        (1, math.inf),
        (1, 0),
        (1, -1),
    ],
)
def test_helper_rejects_nonfinite_or_invalid_parameters(statistic, df):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    with pytest.raises(ValueError):
        two_sided_t_probability(statistic, df)


@pytest.mark.parametrize(
    "df, statistic, expected",
    [
        (2.5, 3.5710801871616802, 0.050109056084874246),
        (2.5, 3.5746548420036839, 0.049999999999999989),
        (2.5, 3.5782294968456871, 0.049891262686851928),
        (7.25, 2.3458466542562668, 0.050175879471896716),
        (7.25, 2.3481948491053721, 0.050000000000000017),
        (7.25, 2.3505430439544774, 0.049824746242553952),
        (11.7, 2.1828401293987461, 0.050194615473990345),
        (11.7, 2.1850251545532995, 0.049999999999999982),
        (11.7, 2.1872101797078525, 0.04980609949468915),
    ],
)
def test_noninteger_tails_near_alpha_match_scipy_reference(df, statistic, expected):
    from insideLLMs.analysis._student_t import two_sided_t_probability

    assert two_sided_t_probability(statistic, df) == pytest.approx(expected, abs=1e-12)


def test_unconverged_fraction_fails_instead_of_returning_partial_probability(monkeypatch):
    import insideLLMs.analysis._student_t as student_t

    # Exercise the real evaluator with a deliberately exhausted iteration budget.
    monkeypatch.setattr(student_t, "_MAX_ITERATIONS", 1)
    with pytest.raises(ArithmeticError, match="did not converge"):
        student_t.two_sided_t_probability(2.0, 30.0)


@pytest.mark.parametrize("value", [-1e-12, 1 + 1e-12, math.nan, math.inf])
def test_materially_invalid_probability_is_not_hidden_by_clamping(value):
    from insideLLMs.analysis._student_t import _checked_probability

    with pytest.raises(ArithmeticError):
        _checked_probability(value)


@pytest.mark.parametrize("value, expected", [(-1e-15, 0), (1 + 1e-15, 1)])
def test_final_roundoff_can_be_clamped(value, expected):
    from insideLLMs.analysis._student_t import _checked_probability

    assert _checked_probability(value) == expected


def test_paired_small_sample_uses_student_t_tail():
    result = paired_t_test([1, 2, 3], [0, 0, 0])
    assert result.p_value == pytest.approx(1 - math.sqrt(6 / 7), abs=1e-12)
    assert result.significant is False


@pytest.mark.parametrize("test", [paired_t_test, welchs_t_test])
def test_single_observation_is_insufficient(test):
    result = test([1], [0])
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.significant is False
    assert "insufficient" in result.conclusion.lower()


@pytest.mark.parametrize("test", [paired_t_test, welchs_t_test])
@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("samples", [[], [1], [1, 2]])
def test_invalid_alpha_is_rejected_even_with_insufficient_data(test, alpha, samples):
    with pytest.raises(ValueError):
        test(samples, samples, alpha=alpha)


@pytest.mark.parametrize("test", [paired_t_test, welchs_t_test])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("samples", [[0], [0, 1]])
def test_nonfinite_observation_is_rejected_in_either_sample(test, value, samples):
    invalid = [value, *samples[1:]]
    with pytest.raises(ValueError):
        test(invalid, samples)
    with pytest.raises(ValueError):
        test(samples, invalid)
