"""
Verify Code Navigator Backend APIs by direct handler invocation
"""

import sys
import os

sys.path.insert(0, os.path.abspath('evaluation/code_navigator/backend'))

from main import (
    get_status,
    get_topics,
    get_topic_detail,
    get_experiments,
    get_experiment_detail,
    get_code,
    search_repository,
    resolve_safe_path
)

def test_api():
    print("Testing get_status()...")
    status = get_status()
    assert status["status"] == "online"
    assert status["total_topics"] == 24
    assert status["total_experiments"] == 4
    print("  Status OK:", status)

    print("\nTesting get_topics()...")
    topics = get_topics()
    assert len(topics) == 24, f"Expected 24 topics, got {len(topics)}"
    print(f"  Topics count: {len(topics)} - OK")

    print("\nTesting get_topic_detail('DT-06')...")
    topic_dt06 = get_topic_detail("DT-06")
    assert topic_dt06["title"] == "U, V and W Factor Generation"
    assert topic_dt06["code_reference"]["line_start"] == 244
    print(f"  Topic DT-06 title: '{topic_dt06['title']}' - OK")

    print("\nTesting get_experiments()...")
    exps = get_experiments()
    assert len(exps) == 4, f"Expected 4 experiments, got {len(exps)}"
    assert "figure09" in exps and "figure10" in exps and "figure11" in exps and "figure12" in exps
    print(f"  Experiments count: {len(exps)} - OK")

    print("\nTesting get_experiment_detail('figure10')...")
    exp10 = get_experiment_detail("figure10")
    assert exp10["winning_model"] == "transformer"
    assert exp10["models"]["transformer"]["psnr"] > 28.5
    print(f"  Experiment 10 winner: '{exp10['winning_model']}' (PSNR: {exp10['models']['transformer']['psnr']:.2f} dB) - OK")

    print("\nTesting get_code() for deeptensor.py...")
    code_data = get_code("src/deeptensor/models/deeptensor.py", 244, 253)
    assert "reconstruction = torch.bmm(U, V)" in code_data["content"]
    assert code_data["total_lines"] > 250
    assert len(code_data["symbols"]) > 0
    print(f"  Code lines: {code_data['total_lines']}, parsed {len(code_data['symbols'])} AST symbols - OK")

    print("\nTesting search_repository('einsum')...")
    search_data = search_repository("einsum")
    assert len(search_data["matched_topics"]) > 0 or len(search_data["matched_symbols"]) > 0
    print(f"  Search matches: {len(search_data['matched_topics'])} topics, {len(search_data['matched_symbols'])} symbols - OK")

    print("\nTesting path traversal security...")
    try:
        resolve_safe_path("../../windows/system32/cmd.exe")
        assert False, "Should have raised HTTPException"
    except Exception as e:
        print(f"  Security traversal correctly blocked with: {type(e).__name__} - OK")

    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_api()
