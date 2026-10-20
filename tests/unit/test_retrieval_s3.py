"""S3-01..05 with stubs (no torch): parser structurally ≡ 0-C index on P-53 (row list, titles), duplicate-id
disambiguation, strict chunk bound, ACL-before-scope, scoped hard filter, grades, open mode returns results but never
grounds ev_AB, C-only scope → grounding none, citation traceability, index/embedder consistency refusals."""
import json
from pathlib import Path
import numpy as np, pytest
from core_pipeline.docs.clause_parser import parse_file, index_rows, compat_rows, structural_diff, parse_docling, Clause
from core_pipeline.docs.chunker import chunk_clause
from core_pipeline.retrieval.acl import load_roles
from core_pipeline.retrieval.index import ChunkIndex
from core_pipeline.retrieval.models import HashEmbedder, LexicalReranker
from core_pipeline.retrieval.retrieve import Retriever
from core_pipeline.retrieval.scope import FaultScope

FIX = Path(__file__).resolve().parents[1] / "fixtures"
P53 = FIX / "P-53-2012.json"; IDX = FIX / "kosha_clause_index.json"; FMAP = FIX / "fault_to_guidance_map.json"
pytestmark = pytest.mark.skipif(not P53.exists(), reason="fixtures missing")


def test_parser_structurally_equivalent_to_0c_on_three_guides():
    rows = json.loads(IDX.read_text(encoding="utf-8"))
    known = set()                                                            # no reviewed exceptions needed any more
    for g, n in (("P-53-2012", 18), ("P-114-2020", 14), ("P-116-2012", 22)):
        idx = [r for r in rows if r["guide_no"] == g]
        cs = parse_file(FIX / f"{g}.json")
        crows, removed = compat_rows(cs)                                     # 0-C-compatible view for the comparison only
        assert len(idx) == n and structural_diff(idx, crows, known) == [], g
        # the corpus view keeps what compat dropped (real sub-headings), and never dropped anything else
        assert sum(int(r["n_items"]) for r in index_rows(cs)) == sum(int(r["n_items"]) for r in crows) + len(removed)
        assert all(r["text"].startswith("(3)") for r in removed)
    # the P-116 appendix case 5 ("5. 2010년 BP Deepwater…", 53 chars) is NOT a clause under the 45-char rule — as in 0-C
    cs = parse_file(FIX / "P-116-2012.json")
    assert not any(c.title.startswith("2010년 BP") for c in cs)



def test_plain_text_numbers_are_not_headers():
    d = {"texts": [{"label": "section_header", "text": "4. 정의", "orig": "4. 정의", "prov": [{"page_no": 1}]},
                   {"label": "text", "text": "0.1 MPa에서 3.5 MPa로 축압될 경우 온도는 약 527 ℃까지 상승한다.", "orig": "0.1 MPa에서 3.5 MPa로 축압될 경우 온도는 약 527 ℃까지 상승한다."},
                   {"label": "list_item", "text": "이 지침에서 사용하는 용어의 뜻은 다음과 같다.", "orig": "3.1 이 지침에서 사용하는 용어의 뜻은 다음과 같다.", "enumerated": True, "marker": "3.1"},
                   {"label": "text", "text": "2.2 저온 고압 산소 벤트", "orig": "2.2 저온 고압 산소 벤트"},
                   {"label": "list_item", "text": "교정일자 : 2000. 00. 00.", "orig": "3. 교정일자 : 2000. 00. 00.", "enumerated": True, "marker": "3."},
                   {"label": "list_item", "text": "10 습식 테스트기 교정", "orig": "10 습식 테스트기 교정", "enumerated": False, "marker": ""},
                   {"label": "text", "text": "200 pF의 정전용량을 가진 사람이 카페트 위를 걸어갈 때 발생되는 정전기는 약 10,000 V인 경우의 정전기에너지는 얼마인가?", "orig": "200 pF의 정전용량을 가진 사람이 카페트 위를 걸어갈 때 발생되는 정전기는 약 10,000 V인 경우의 정전기에너지는 얼마인가?"},
                   {"label": "list_item", "text": "습도 조절 및 불활성가스 등에 의한 방법", "orig": "6.3 습도 조절 및 불활성가스 등에 의한 방법", "enumerated": True, "marker": "6.3"},
                   {"label": "section_header", "text": "5.5 . 보안(Security)", "orig": "5.5 . 보안(Security)", "prov": [{"page_no": 2}]},
                   {"label": "section_header", "text": "5.3.7.3.1 시험전의 준비", "orig": "5.3.7.3.1 시험전의 준비", "prov": [{"page_no": 2}]}]}
    cs = parse_docling(d, "P-T-2000")
    assert [(c.clause_id, c.title) for c in cs] == [("4", "정의"), ("2.2", "저온 고압 산소 벤트"), ("3", "교정일자 : 2000. 00. 00."),
                                                     ("10", "습식 테스트기 교정"),
                                                     ("6.3", "습도 조절 및 불활성가스 등에 의한 방법"), ("5.5", "보안(Security)")]
    assert cs[3].items == ["200 pF의 정전용량을 가진 사람이 카페트 위를 걸어갈 때 발생되는 정전기는 약 10,000 V인 경우의 정전기에너지는 얼마인가?"]
    assert cs[0].items == ["0.1 MPa에서 3.5 MPa로 축압될 경우 온도는 약 527 ℃까지 상승한다.", "이 지침에서 사용하는 용어의 뜻은 다음과 같다."]
    assert cs[-1].items == ["5.3.7.3.1 시험전의 준비"]                    # depth-5 numbering is an item (0-C cap at 4)


def test_duplicate_ids_get_occurrence_suffix_and_0c_index_has_the_duplicate():
    rows = json.loads(IDX.read_text(encoding="utf-8"))
    keys = [r["clause_key"] for r in rows]
    assert len(keys) == 699 and len(set(keys)) == 698 and keys.count("P-132-2013:0:4.8") == 2
    d = {"texts": [{"label": "section_header", "text": "1. 목적", "prov": [{"page_no": 1}]}, {"label": "text", "text": "x"},
                   {"label": "section_header", "text": "4.8 교반기 정지", "prov": [{"page_no": 15}]}, {"label": "text", "text": "y"},
                   {"label": "section_header", "text": "4.8 미반응 물질의 축적", "prov": [{"page_no": 16}]}, {"label": "text", "text": "z"}]}
    cs = parse_docling(d, "P-132-2013")
    assert [c.clause_key for c in cs] == ["P-132-2013:0:1", "P-132-2013:0:4.8~1", "P-132-2013:0:4.8~2"]
    assert [c.source_key for c in cs][1:] == ["P-132-2013:0:4.8"] * 2        # structural check still sees the bare key
    fm = json.loads(FMAP.read_text(encoding="utf-8"))
    assert "P-132-2013:0:4.8~2" in fm["faults"]["1"]["clauses"]["B"] and "P-132-2013:0:4.8" not in fm["faults"]["1"]["clauses"]["B"]
    assert fm["amendments"][0]["decision"] == "ST3-05"


def test_chunking_strict_bound():
    long_item = ("문장입니다. " * 150).strip()                               # 1049 chars, sentence boundaries available
    c = Clause("P-X-2000", 0, "4", "t", 1, 3, items=["a" * 600, "b" * 600, "c" * 50, "KOSHA GUIDE", long_item, "q" * 1500])
    ch = chunk_clause(c)
    assert all(len(x.text) <= 1000 for x in ch)                              # strict, separators included
    assert ch[0].chunk_key == "P-X-2000:0:4#0" and all(x.n_chunks == len(ch) for x in ch)
    assert "KOSHA GUIDE" not in "".join(x.text for x in ch)
    # no text is lost by splitting: concatenated chunk text == concatenated (header-stripped) items
    kept = [t for t in c.items if t != "KOSHA GUIDE"]
    assert "".join(x.text.replace("\n", "") for x in ch) == "".join(kept).replace(" ", "").replace("\n", "") or \
           "".join(x.text for x in ch).replace("\n", "") == "".join(kept)
    two = Clause("P-X-2000", 0, "6", "t", 1, 3, items=["x" * 500, "y" * 500])  # 500+1+500 = 1001 > 1000 → two chunks
    assert len(chunk_clause(two)) == 2
    small = Clause("P-X-2000", 0, "5", "t", 1, 3, items=["x" * 100])
    assert chunk_clause(small)[0].chunk_key == "P-X-2000:0:5" and chunk_clause(small)[0].n_chunks == 1


@pytest.fixture(scope="module")
def retr():
    clauses = parse_file(P53)
    # a fake distractor guide so ACL has something to cut, plus a common clause lookalike
    fake = Clause("P-44-2012", 0, "3", "응축기 냉각 일반", 1, 2, items=["응축기 냉각수 공급 중단 시 추가 냉각과 반응속도 제어가 필요하다."])
    common = Clause("P-116-2012", 0, "5.2.4", "경보 설정값의 결정", 3, 9, items=["경보 설정값은 공정 변수의 정상 범위와 안전 한계를 고려하여 결정한다."])
    chunks = [ch.to_dict() for c in clauses + [fake, common] for ch in chunk_clause(c)]
    idx = ChunkIndex.build(chunks, HashEmbedder())
    scope = FaultScope(FMAP)
    roles = load_roles(FMAP, {c["guide_no"] for c in chunks})
    return Retriever(idx, HashEmbedder(), LexicalReranker(), scope, roles, tau=None)


def test_scoped_is_hard_filter_and_grades(retr):
    r = retr("응축기 냉각수 온도 상승 시 추가 냉각 조치", pred_set=[5], caller_role="engineer", mode="scoped")
    assert r.status == "OK" and r.chunks and r.grounding == "unknown_tau"
    keys = retr.scope.scope_keys([5])
    assert all(c.clause_key in keys for c in r.chunks)                      # nothing outside A∪B∪C of fault 5
    assert all(c.grade in ("A", "B", "C") for c in r.chunks)
    assert any(c.clause_key == "P-53-2012:0:6.5" and c.grade == "A" for c in r.chunks)   # 추가 냉각 is A for fault 5
    assert r.ev_AB is None and r.tau is None                                 # tau not fixed yet → policy cannot propose


def test_open_mode_returns_results_but_never_grounds(retr):
    retr.tau = 0.0                                                           # permissive tau on the [0,1] scale
    r = retr("응축기 냉각", pred_set=[5], caller_role="engineer", mode="open")
    assert r.status == "OK" and r.chunks                                     # results ARE returned (the bug the reviewer caught)
    assert r.ev_AB is False and r.grounding in ("C_only", "none")            # …but never qualify as proposal grounding
    assert all(0.0 <= c.score <= 1.0 for c in r.chunks)                      # reranker scale contract
    assert any(c.grade == "OUT" for c in r.chunks) or r.scope_size > len(retr.scope.scope_keys([5]))
    r2 = retr("응축기 냉각", pred_set=[5], caller_role="restricted", mode="open")
    assert all(c.guide_no != "P-44-2012" for c in r2.chunks)                 # distractor removed by ACL before ranking
    retr.tau = None


def test_c_only_scope_cannot_ground(retr):
    retr.tau = 0.0
    r = retr("경보 설정값 결정", pred_set=[18], caller_role="engineer", mode="scoped")   # faults 16–20: C only
    assert r.status == "OK" and all(c.grade == "C" for c in r.chunks) and r.g_AB is None
    assert r.ev_AB is False and r.grounding == "C_only"
    r2 = retr("경보 설정값 결정", pred_set=[18], caller_role="restricted", mode="scoped")
    assert r2.status in ("OK", "NO_RESULTS")                                 # restricted keeps mapped guides, so may still hit
    empty = retr("아무거나", pred_set=[], caller_role="restricted", mode="scoped")
    assert empty.status in ("OK", "NO_RESULTS") and empty.ev_AB is False     # empty set → C only; never grounds
    retr.tau = None


def test_index_consistency_refusals(tmp_path, retr):
    idx = retr.index
    idx.save(tmp_path)
    ChunkIndex.load(tmp_path)                                                # clean round trip
    (tmp_path / "chunks.jsonl").write_text((tmp_path / "chunks.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ChunkIndex.load(tmp_path)                                            # chunk file changed under the vectors
    class OtherEmbedder(HashEmbedder):
        name = "other-embedder"
    with pytest.raises(ValueError):
        Retriever(idx, OtherEmbedder(), LexicalReranker(), retr.scope, retr.roles)   # query embedder ≠ index embedder


def test_every_result_is_a_real_chunk(retr):
    r = retr("폭주반응 긴급배출", pred_set=[4], caller_role="engineer", mode="scoped", k=10)
    keys = {c["chunk_key"]: c for c in retr.index.chunks}
    assert all(c.chunk_key in keys and keys[c.chunk_key]["text"] == c.text for c in r.chunks)
    assert [c.rank for c in r.chunks] == list(range(1, len(r.chunks) + 1))
