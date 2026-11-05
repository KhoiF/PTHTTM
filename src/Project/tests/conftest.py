import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _services_up() -> bool:
    try:
        import ollama
        from kb.load_neo4j import get_driver

        ollama.Client().list()
        with get_driver() as driver:
            driver.verify_connectivity()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def fixture_kg():
    """Nạp đồ thị mẫu (tests/fixtures/sample_graph.json) vào nhãn DKGTEST, xóa sau khi chạy xong."""
    if not _services_up():
        pytest.skip("cần Ollama và Neo4j đang chạy")
    import config
    from agents.llm import LLM
    from kb import load_neo4j

    old_label = config.KG_LABEL
    config.KG_LABEL = "DKGTEST"
    data = json.loads((FIXTURES / "sample_graph.json").read_text(encoding="utf-8"))
    with load_neo4j.get_driver() as driver:
        load_neo4j.reset(driver)
        load_neo4j.ensure_schema(driver)
        load_neo4j.load_graph(driver, data["graph"], data["chunks"], [data["page"]], LLM())
    yield data
    with load_neo4j.get_driver() as driver:
        load_neo4j.reset(driver)
    config.KG_LABEL = old_label
