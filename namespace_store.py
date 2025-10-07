"""
namespace_store.py
------------------
Small JSON-backed store for previously used Pinecone namespaces.
"""
import json, os
from typing import List

_NS_FILE = "./.ns_history.json"

def _load() -> List[str]:
    if not os.path.exists(_NS_FILE):
        return []
    try:
        with open(_NS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [n for n in data if isinstance(n, str)]
    except Exception:
        return []

def _save(namespaces: List[str]) -> None:
    uniq = sorted(set([n for n in namespaces if n]))
    with open(_NS_FILE, "w", encoding="utf-8") as f:
        json.dump(uniq, f, indent=2)

def add_namespace(ns: str) -> None:
    if not ns: return
    ns_list = _load()
    if ns not in ns_list:
        ns_list.append(ns)
        _save(ns_list)

def list_namespaces() -> List[str]:
    return _load()

def delete_namespace(ns: str) -> None:
    lst = _load()
    lst = [x for x in lst if x != ns]
    _save(lst)