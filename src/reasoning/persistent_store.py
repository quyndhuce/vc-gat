"""
Persistent Key-Value / B-Tree Storage Engine on NVMe SSD (Section 6.2 & Appendix C.3).
"""

import dbm
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class PersistentDerivationsStore:
    def __init__(self, db_path: str, mode: str = "r"):
        self.db_path = str(db_path)
        self.mode = mode
        self._db = None

    def __enter__(self):
        self._db = dbm.open(self.db_path, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._db is not None:
            self._db.close()
            self._db = None

    def put(self, key: str, derivations: List[Any]) -> None:
        if self._db is None:
            raise RuntimeError("Database not open.")
        self._db[key.encode("utf-8")] = json.dumps(derivations).encode("utf-8")

    def get(self, key: str) -> Optional[List[Any]]:
        if self._db is None:
            raise RuntimeError("Database not open.")
        k_bytes = key.encode("utf-8")
        if k_bytes in self._db:
            return json.loads(self._db[k_bytes].decode("utf-8"))
        return None

    def contains(self, key: str) -> bool:
        if self._db is None:
            raise RuntimeError("Database not open.")
        return key.encode("utf-8") in self._db
