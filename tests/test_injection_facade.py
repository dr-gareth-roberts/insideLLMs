"""Compatibility contract for the public injection facade."""


def test_injection_facade_reexports_engine_symbols() -> None:
    import insideLLMs.injection as facade
    import insideLLMs.security.injection_engine as engine

    assert facade.InjectionDetector is engine.InjectionDetector
    assert facade.InputSanitizer is engine.InputSanitizer
    assert facade.DefensivePromptBuilder is engine.DefensivePromptBuilder
    assert facade.InjectionTester is engine.InjectionTester
    assert facade.detect_injection is engine.detect_injection
    assert facade.sanitize_input is engine.sanitize_input
    assert facade.build_defensive_prompt is engine.build_defensive_prompt
