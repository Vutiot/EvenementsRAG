#!/usr/bin/env python3
"""One-time migration: alias the unified collection name to the legacy collection.

Creates a Qdrant alias so that the new canonical name
``wiki_10k_qdrant_cs512_co50_minilm_l6_cosine`` resolves to the existing
``ww2_events_10000`` collection without re-indexing.

Usage:
    .venv/bin/python scripts/migrate_collection_alias.py
"""

from __future__ import annotations

import sys

from qdrant_client import QdrantClient, models

OLD_NAME = "ww2_events_10000"
NEW_NAME = "wiki_10k_qdrant_cs512_co50_minilm_l6_cosine"


def main() -> None:
    from config.settings import settings

    location = settings.QDRANT_LOCATION
    if isinstance(location, tuple):
        client = QdrantClient(host=location[0], port=location[1])
    else:
        client = QdrantClient(url=location)

    # Check if old collection exists
    existing = [c.name for c in client.get_collections().collections]
    if OLD_NAME not in existing:
        print(f"Collection '{OLD_NAME}' not found — nothing to migrate.")
        sys.exit(0)

    # Check if new name already exists as a collection or alias
    if NEW_NAME in existing:
        print(f"'{NEW_NAME}' already exists as a collection — skipping alias.")
        sys.exit(0)

    aliases = client.get_collection_aliases(OLD_NAME).aliases
    if any(a.alias_name == NEW_NAME for a in aliases):
        print(f"Alias '{NEW_NAME}' already exists — nothing to do.")
        sys.exit(0)

    # Create alias
    client.update_collection_aliases(
        change_aliases_operations=[
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=OLD_NAME,
                    alias_name=NEW_NAME,
                )
            )
        ]
    )
    print(f"Created alias '{NEW_NAME}' -> '{OLD_NAME}'")


if __name__ == "__main__":
    main()
