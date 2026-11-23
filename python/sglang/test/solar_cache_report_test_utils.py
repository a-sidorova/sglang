"""Shared cache-report cases for Solar's Chat and Completions tests."""

from sglang.srt.runtime_context import get_context


def assert_cache_report_usage(test_case, spec_result, build_usage):
    for enabled in (False, True):
        for cached_tokens in (None, 0, 6):
            for stream in (False, True):
                with (
                    test_case.subTest(
                        enabled=enabled, cached_tokens=cached_tokens, stream=stream
                    ),
                    get_context().override_server_args(enable_cache_report=enabled),
                ):
                    content = spec_result(0)
                    if cached_tokens is None:
                        content["meta_info"].pop("cached_tokens")
                    else:
                        content["meta_info"]["cached_tokens"] = cached_tokens
                    usage = build_usage(content, stream)
                    expected = (
                        {"cached_tokens": cached_tokens or 0} if enabled else None
                    )
                    test_case.assertEqual(usage.get("prompt_tokens_details"), expected)


def single_stream_usage(test_case, chunks):
    usages = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
    test_case.assertEqual(len(usages), 1)
    return usages[0]
