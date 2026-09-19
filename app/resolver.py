"""跨文档 ``$ref`` 引用解析：``doc:<ID>@<正整数|current>#<RFC6901指针>``。

规则概述
--------
* 仅当对象 **只含** ``$ref`` 一个键、且值为匹配
  ``doc:<ID>@<正整数|current>#<RFC6901指针>`` 的字符串时视为引用；
  其余对象（``$ref`` 附带其它键、值形态不符）按普通对象处理，原样保留；
* 解析递归展开对象与数组中的引用；被引用子树内的引用一并展开；
* 同一 ``(document_id, revision)`` 快照在解析栈上重复出现 → 依赖成环；
* 自根内容起跟随引用的跳数上限为 :data:`MAX_DEPTH`，超出即失败；
* 依赖清单按首次遍历顺序去重：同一快照被多次引用只记录一次。

本模块不依赖 FastAPI / SQLite：文档快照通过回调注入（调用方负责让两个
回调在同一个读快照内作答），可独立单测。
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

#: 引用嵌套深度上限（自根内容起跟随的引用跳数；根内容本身深度为 0）
MAX_DEPTH = 32

#: 引用值形态：doc:<ID>@<正整数|current>#<RFC6901 指针>。
#: ID 字符集与文档 ID 的创建约束一致；修订号为不带前导零的正整数或 current。
REF_RE = re.compile(
    r"^doc:(?P<id>[A-Za-z0-9][A-Za-z0-9._~-]*)"
    r"@(?P<revision>[1-9][0-9]*|current)"
    r"#(?P<pointer>.*)$"
)


class ResolutionError(Exception):
    """解析失败：携带 HTTP 状态码、错误码与完整引用链（含失败一跳）。"""

    def __init__(
        self,
        status_code: int,
        code: str,
        chain: list[dict[str, Any]],
        **extra: Any,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.chain = chain
        self.extra = extra
        super().__init__(code)


def match_reference(value: Any) -> "Optional[re.Match[str]]":
    """若 *value* 是引用对象（仅含 ``$ref`` 且形态合法），返回正则匹配。"""
    if (
        isinstance(value, dict)
        and len(value) == 1
        and isinstance(value.get("$ref"), str)
    ):
        return REF_RE.match(value["$ref"])
    return None


def _hop(document_id: str, revision: Optional[int], pointer: str) -> dict[str, Any]:
    """引用链上的一跳：在 *document_id* 的 *revision* 快照内定位到 *pointer*。"""
    return {"document_id": document_id, "revision": revision, "pointer": pointer}


def _parse_pointer(pointer: str) -> Optional[list[str]]:
    """按 RFC 6901 把指针解码为 token 序列；语法非法（非空且不以 ``/`` 开头）返回 None。"""
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        return None
    # RFC 6901 §4：先把 ~1 还原为 /，再把 ~0 还原为 ~
    return [token.replace("~1", "/").replace("~0", "~") for token in pointer.split("/")[1:]]


class _PointerMiss(Exception):
    """指针在目标快照中不存在（内部控制流，翻译为 422）。"""


def _navigate(content: Any, tokens: list[str]) -> Any:
    """沿 *tokens* 在 *content* 内定位；任一级缺失抛 :class:`_PointerMiss`。"""
    node = content
    for token in tokens:
        if isinstance(node, dict):
            if token not in node:
                raise _PointerMiss
            node = node[token]
        elif isinstance(node, list):
            # 数组下标：不带前导零的非负整数，且必须在界内
            if (
                not token.isdigit()
                or (token != "0" and token.startswith("0"))
                or int(token) >= len(node)
            ):
                raise _PointerMiss
            node = node[int(token)]
        else:
            raise _PointerMiss
    return node


def resolve(
    root_id: str,
    root_revision: int,
    root_content: Any,
    get_current_revision: Callable[[str], Optional[int]],
    get_snapshot: Callable[[str, int], Optional[Any]],
) -> "tuple[Any, list[dict[str, Any]]]":
    """递归展开 *root_content* 中的引用，返回 ``(resolved, dependencies)``。

    :param get_current_revision: 返回文档当前修订号，文档不存在返回 None；
    :param get_snapshot: 返回指定修订的内容，修订不存在返回 None。
    两个回调必须在同一读快照内作答（由调用方的事务保证），这样所有
    ``current`` 引用与根文档修订在整个解析过程中保持固定。

    依赖清单按首次遍历（深度优先）顺序去重，元素形如
    ``{"document_id": ..., "revision": ...}``；根文档不计入依赖。

    :raises ResolutionError: 引用的文档 / 修订 / 指针不存在（422）、
        依赖成环（409）、嵌套深度超过 :data:`MAX_DEPTH`（422）。
    """
    dependencies: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def record(document_id: str, revision: int) -> None:
        key = (document_id, revision)
        if key not in seen:
            seen.add(key)
            dependencies.append({"document_id": document_id, "revision": revision})

    def resolve_node(node: Any, chain: list[dict[str, Any]], depth: int) -> Any:
        match = match_reference(node)
        if match is None:
            if isinstance(node, dict):
                return {
                    key: resolve_node(value, chain, depth)
                    for key, value in node.items()
                }
            if isinstance(node, list):
                return [resolve_node(item, chain, depth) for item in node]
            return node

        document_id = match["id"]
        revision_spec = match["revision"]
        pointer = match["pointer"]
        ref = match.group(0)
        # 形如 @5 时请求的具体修订号；@current 时尚未固定（查表后才知道）
        requested = None if revision_spec == "current" else int(revision_spec)

        # 深度检查先于任何读库：链再长也不做无谓的快照读取
        if depth >= MAX_DEPTH:
            raise ResolutionError(
                422,
                "reference_depth_exceeded",
                chain + [_hop(document_id, requested, pointer)],
                ref=ref,
                max_depth=MAX_DEPTH,
                reason=f"引用嵌套深度超过 {MAX_DEPTH}",
            )

        current = get_current_revision(document_id)
        if current is None:
            raise ResolutionError(
                422,
                "reference_document_not_found",
                chain + [_hop(document_id, requested, pointer)],
                ref=ref,
                reason="引用的文档不存在",
            )
        revision = current if requested is None else requested

        snapshot = get_snapshot(document_id, revision)
        if snapshot is None:
            raise ResolutionError(
                422,
                "reference_revision_not_found",
                chain + [_hop(document_id, revision, pointer)],
                ref=ref,
                reason="引用的修订不存在",
            )

        # 依赖成环：同一 (document_id, revision) 快照在解析栈上重复出现。
        # 环链裁剪到首次出现处，使首尾为同一快照（首尾闭合）。
        hop = _hop(document_id, revision, pointer)
        for index, entry in enumerate(chain):
            if entry["document_id"] == document_id and entry["revision"] == revision:
                raise ResolutionError(
                    409,
                    "reference_cycle",
                    chain[index:] + [hop],
                    ref=ref,
                    reason="依赖成环",
                )

        try:
            tokens = _parse_pointer(pointer)
            if tokens is None:
                raise _PointerMiss
            target = _navigate(snapshot, tokens)
        except _PointerMiss:
            raise ResolutionError(
                422,
                "reference_pointer_not_found",
                chain + [hop],
                ref=ref,
                reason="引用的 JSON 指针在目标快照中不存在",
            ) from None

        record(document_id, revision)
        return resolve_node(target, chain + [hop], depth + 1)

    resolved = resolve_node(root_content, [_hop(root_id, root_revision, "")], 0)
    return resolved, dependencies
