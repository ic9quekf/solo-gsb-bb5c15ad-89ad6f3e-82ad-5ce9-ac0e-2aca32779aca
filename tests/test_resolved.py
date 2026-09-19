"""GET /documents/{id}/resolved 跨文档引用解析端到端集成测试（独立临时数据库）。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 与 test_api 相同：直接改 settings.db_path，避免模块单例身份漂移
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, content=None, doc_id="cfg"):
    body = {"id": doc_id}
    if content is not None:
        body["content"] = content
    r = client.post("/documents", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def _patch(client, doc_id, base_revision, request_id, patch):
    r = client.patch(
        f"/documents/{doc_id}",
        json={
            "base_revision": base_revision,
            "request_id": request_id,
            "patch": patch,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def _resolved(client, doc_id, revision=None):
    params = {} if revision is None else {"revision": revision}
    return client.get(f"/documents/{doc_id}/resolved", params=params)


# ---------------- 基本解析 ----------------


def test_no_references_returns_content_and_empty_dependencies(client):
    _create(client, {"a": 1, "nested": {"x": [1, 2]}}, doc_id="plain")
    r = _resolved(client, "plain")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "plain"
    assert body["revision"] == 1
    assert body["resolved"] == {"a": 1, "nested": {"x": [1, 2]}}
    assert body["dependencies"] == []


def test_resolves_current_reference_and_pins_revision(client):
    _create(client, {"db": {"host": "a.internal", "port": 5432}}, doc_id="infra")
    _patch(client, "infra", 1, "r1", {"db": {"port": 5433}})  # current = 2
    _create(client, {"settings": {"$ref": "doc:infra@current#/db"}}, doc_id="app")
    r = _resolved(client, "app")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {"settings": {"host": "a.internal", "port": 5433}}
    assert body["dependencies"] == [{"document_id": "infra", "revision": 2}]


def test_resolves_pinned_revision_reference(client):
    _create(client, {"v": 1}, doc_id="lib")
    _patch(client, "lib", 1, "r1", {"v": 2})  # rev2
    _create(client, {"locked": {"$ref": "doc:lib@1#"}}, doc_id="app")
    body = _resolved(client, "app").json()
    assert body["resolved"] == {"locked": {"v": 1}}
    assert body["dependencies"] == [{"document_id": "lib", "revision": 1}]


def test_empty_pointer_resolves_whole_document(client):
    _create(client, {"x": 1, "y": {"z": 2}}, doc_id="whole")
    _create(client, {"copy": {"$ref": "doc:whole@1#"}}, doc_id="app")
    body = _resolved(client, "app").json()
    assert body["resolved"] == {"copy": {"x": 1, "y": {"z": 2}}}


def test_nested_reference_chain(client):
    _create(client, {"leaf": 42}, doc_id="c")
    _create(client, {"mid": {"$ref": "doc:c@1#/leaf"}}, doc_id="b")
    _create(client, {"top": {"$ref": "doc:b@1#/mid"}}, doc_id="a")
    body = _resolved(client, "a").json()
    assert body["resolved"] == {"top": 42}
    # 依赖按首次遍历顺序：先 b 后 c
    assert body["dependencies"] == [
        {"document_id": "b", "revision": 1},
        {"document_id": "c", "revision": 1},
    ]


def test_references_inside_arrays(client):
    _create(client, {"n": 1}, doc_id="one")
    _create(client, {"n": 2}, doc_id="two")
    _create(
        client,
        {"items": [{"$ref": "doc:one@1#/n"}, {"$ref": "doc:two@1#/n"}, 3]},
        doc_id="arr",
    )
    body = _resolved(client, "arr").json()
    assert body["resolved"] == {"items": [1, 2, 3]}
    assert body["dependencies"] == [
        {"document_id": "one", "revision": 1},
        {"document_id": "two", "revision": 1},
    ]


def test_reference_to_scalar_and_array_values(client):
    _create(client, {"list": [1, 2, 3], "s": "txt"}, doc_id="vals")
    _create(
        client,
        {"a": {"$ref": "doc:vals@1#/list"}, "b": {"$ref": "doc:vals@1#/s"}},
        doc_id="app",
    )
    body = _resolved(client, "app").json()
    assert body["resolved"] == {"a": [1, 2, 3], "b": "txt"}
    # 同一快照被多次引用只记录一次
    assert body["dependencies"] == [{"document_id": "vals", "revision": 1}]


def test_diamond_dependencies_deduped_in_first_traversal_order(client):
    _create(client, {"base": True}, doc_id="d")
    _create(client, {"from_b": {"$ref": "doc:d@1#/base"}}, doc_id="b")
    _create(client, {"from_c": {"$ref": "doc:d@1#/base"}}, doc_id="c")
    _create(
        client,
        {"x": {"$ref": "doc:b@1#/from_b"}, "y": {"$ref": "doc:c@1#/from_c"}},
        doc_id="a",
    )
    body = _resolved(client, "a").json()
    assert body["resolved"] == {"x": True, "y": True}
    # 深度优先首次遍历：b → d → c（d 在 b 子树中已记录，c 子树中不再重复）
    assert body["dependencies"] == [
        {"document_id": "b", "revision": 1},
        {"document_id": "d", "revision": 1},
        {"document_id": "c", "revision": 1},
    ]


def test_same_document_at_different_revisions_listed_separately(client):
    _create(client, {"v": 1}, doc_id="lib")
    _patch(client, "lib", 1, "r1", {"v": 2})  # rev2
    _create(
        client,
        {"a_old": {"$ref": "doc:lib@1#/v"}, "b_new": {"$ref": "doc:lib@2#/v"}},
        doc_id="app",
    )
    body = _resolved(client, "app").json()
    assert body["resolved"] == {"a_old": 1, "b_new": 2}
    assert body["dependencies"] == [
        {"document_id": "lib", "revision": 1},
        {"document_id": "lib", "revision": 2},
    ]


def test_root_revision_query_param_pins_root(client):
    _create(client, {"v": "old"}, doc_id="dep")
    _create(client, {"x": {"$ref": "doc:dep@1#/v"}}, doc_id="app")
    _patch(client, "app", 1, "r1", {"x": None, "y": "no-refs"})  # rev2 移除引用
    r = _resolved(client, "app", revision=1)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["revision"] == 1
    assert body["resolved"] == {"x": "old"}
    assert body["dependencies"] == [{"document_id": "dep", "revision": 1}]
    # 默认解析当前版（rev2，无引用）
    body2 = _resolved(client, "app").json()
    assert body2["revision"] == 2
    assert body2["resolved"] == {"y": "no-refs"}
    assert body2["dependencies"] == []


def test_pointer_escapes_and_array_index(client):
    _create(client, {"a/b": {"~key": ["zero", "one"]}}, doc_id="esc")
    _create(
        client,
        {
            "slash": {"$ref": "doc:esc@1#/a~1b/~0key/0"},
            "tilde": {"$ref": "doc:esc@1#/a~1b/~0key/1"},
        },
        doc_id="app",
    )
    body = _resolved(client, "app").json()
    assert body["resolved"] == {"slash": "zero", "tilde": "one"}


def test_non_reference_ref_objects_left_as_is(client):
    _create(client, {"v": 1}, doc_id="lib")
    content = {
        "extra_key": {"$ref": "doc:lib@1#/v", "other": 1},  # $ref 附带其它键
        "not_doc": {"$ref": "http://example.com/x"},  # 非 doc: 形态
        "no_pointer": {"$ref": "doc:lib@1"},  # 缺 # 指针
        "bad_rev": {"$ref": "doc:lib@0#/v"},  # 非正整数修订
        "not_string": {"$ref": 42},  # 值不是字符串
    }
    _create(client, content, doc_id="app")
    body = _resolved(client, "app").json()
    assert body["resolved"] == content
    assert body["dependencies"] == []


def test_resolution_does_not_modify_stored_content(client):
    _create(client, {"v": 1}, doc_id="lib")
    _create(client, {"x": {"$ref": "doc:lib@1#/v"}}, doc_id="app")
    _resolved(client, "app")
    raw = client.get("/documents/app").json()
    assert raw["content"] == {"x": {"$ref": "doc:lib@1#/v"}}


# ---------------- 根文档 / 根修订缺失 ----------------


def test_missing_root_document_returns_404(client):
    r = _resolved(client, "ghost")
    assert r.status_code == 404
    assert r.json()["detail"]["error"] == "not_found"


def test_missing_root_revision_returns_422_with_chain(client):
    _create(client, {"a": 1}, doc_id="app")
    r = _resolved(client, "app", revision=99)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "revision_not_found"
    assert detail["chain"] == [
        {"document_id": "app", "revision": 99, "pointer": ""}
    ]


def test_invalid_revision_query_param_rejected(client):
    _create(client, {"a": 1}, doc_id="app")
    assert _resolved(client, "app", revision=0).status_code == 422
    assert _resolved(client, "app", revision=-1).status_code == 422


# ---------------- 引用目标缺失：422 + 完整引用链 ----------------


def test_missing_referenced_document_422_with_chain(client):
    _create(client, {"x": {"$ref": "doc:ghost@current#/y"}}, doc_id="app")
    r = _resolved(client, "app")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "reference_document_not_found"
    assert detail["ref"] == "doc:ghost@current#/y"
    assert detail["chain"] == [
        {"document_id": "app", "revision": 1, "pointer": ""},
        {"document_id": "ghost", "revision": None, "pointer": "/y"},
    ]


def test_missing_referenced_revision_422_with_chain(client):
    _create(client, {"v": 1}, doc_id="lib")
    _create(client, {"x": {"$ref": "doc:lib@5#/v"}}, doc_id="app")
    r = _resolved(client, "app")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "reference_revision_not_found"
    assert detail["chain"] == [
        {"document_id": "app", "revision": 1, "pointer": ""},
        {"document_id": "lib", "revision": 5, "pointer": "/v"},
    ]


def test_missing_pointer_422_with_full_chain(client):
    _create(client, {"leaf": 1}, doc_id="c")
    _create(client, {"mid": {"$ref": "doc:c@1#/nope"}}, doc_id="b")
    _create(client, {"top": {"$ref": "doc:b@1#/mid"}}, doc_id="a")
    r = _resolved(client, "a")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "reference_pointer_not_found"
    assert detail["ref"] == "doc:c@1#/nope"
    # 完整引用链：根 → b → 失败一跳
    assert detail["chain"] == [
        {"document_id": "a", "revision": 1, "pointer": ""},
        {"document_id": "b", "revision": 1, "pointer": "/mid"},
        {"document_id": "c", "revision": 1, "pointer": "/nope"},
    ]


def test_pointer_miss_through_array_and_scalar(client):
    _create(client, {"arr": [1], "s": "x"}, doc_id="lib")
    _create(
        client,
        {
            "bad_index": {"$ref": "doc:lib@1#/arr/5"},
            "into_scalar": {"$ref": "doc:lib@1#/s/deep"},
        },
        doc_id="app",
    )
    r = _resolved(client, "app")
    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "reference_pointer_not_found"


# ---------------- 依赖成环：409 + 首尾闭合链 ----------------


def test_direct_cycle_between_two_documents(client):
    _create(client, {"to_b": {"$ref": "doc:b@current#/to_a"}}, doc_id="a")
    _create(client, {"to_a": {"$ref": "doc:a@current#/to_b"}}, doc_id="b")
    r = _resolved(client, "a")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "reference_cycle"
    chain = detail["chain"]
    # 首尾闭合：链首与链尾是同一 (document_id, revision) 快照
    assert chain[0]["document_id"] == chain[-1]["document_id"]
    assert chain[0]["revision"] == chain[-1]["revision"]
    assert [hop["document_id"] for hop in chain] == ["a", "b", "a"]


def test_self_reference_is_cycle(client):
    _create(client, {"x": 1, "self": {"$ref": "doc:loop@current#/x"}}, doc_id="loop")
    r = _resolved(client, "loop")
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "reference_cycle"
    chain = detail["chain"]
    assert [hop["document_id"] for hop in chain] == ["loop", "loop"]
    assert chain[0]["revision"] == chain[-1]["revision"] == 1


def test_indirect_cycle_chain_trimmed_to_closed_loop(client):
    _create(client, {"to_c": {"$ref": "doc:c@1#/to_b"}}, doc_id="b")
    _create(client, {"to_b": {"$ref": "doc:b@1#/to_c"}}, doc_id="c")
    _create(client, {"start": {"$ref": "doc:b@1#/to_c"}}, doc_id="root")
    r = _resolved(client, "root")
    assert r.status_code == 409
    chain = r.json()["detail"]["chain"]
    # 环链裁剪到环本身（root 不在环上，不出现在链首）
    assert [hop["document_id"] for hop in chain] == ["b", "c", "b"]
    assert chain[0]["revision"] == chain[-1]["revision"]


def test_same_document_different_revisions_is_not_a_cycle(client):
    # lib@3 → lib@2 → lib@1：同一文档不同修订不构成环
    _create(client, {"v": "one"}, doc_id="lib")
    _patch(client, "lib", 1, "r1", {"v": "two", "prev": {"$ref": "doc:lib@1#/v"}})
    _patch(client, "lib", 2, "r2", {"v": "three", "prev": {"$ref": "doc:lib@2#/prev"}})
    r = _resolved(client, "lib", revision=3)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["resolved"] == {"v": "three", "prev": "one"}
    assert body["dependencies"] == [
        {"document_id": "lib", "revision": 2},
        {"document_id": "lib", "revision": 1},
    ]


# ---------------- 深度上限 ----------------


def _build_chain(client, links):
    """创建 d0 → d1 → … → d{links} 的引用链（d{links} 为静态叶子）。"""
    _create(client, {"next": "leaf"}, doc_id=f"d{links}")
    for i in range(links - 1, -1, -1):
        _create(client, {"next": {"$ref": f"doc:d{i + 1}@1#/next"}}, doc_id=f"d{i}")


def test_depth_up_to_32_references_resolves(client):
    _build_chain(client, 32)  # 根 + 32 跳引用
    r = _resolved(client, "d0")
    assert r.status_code == 200, r.text
    assert r.json()["resolved"] == {"next": "leaf"}
    assert len(r.json()["dependencies"]) == 32


def test_depth_beyond_32_references_rejected(client):
    _build_chain(client, 33)  # 第 33 跳超出上限
    r = _resolved(client, "d0")
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["error"] == "reference_depth_exceeded"
    assert detail["max_depth"] == 32
    assert len(detail["chain"]) == 34  # 根 + 32 跳 + 失败一跳
