"""
namespace_admin.py
------------------
Admin helpers for Pinecone namespaces (delete/purge).
"""
from utils.config import index

def purge_namespace(ns: str):
    """
    Delete ALL vectors in the namespace. Irreversible.
    """
    index.delete(namespace=ns, delete_all=True)