from qdrant_client import QdrantClient

# ====== CONFIGURATION ======
QDRANT_HOST = "localhost"   # Change if needed
QDRANT_PORT = 7333          # Default REST port
USE_HTTPS = False           # Set True if using cloud with https
API_KEY = None              # Add API key if using Qdrant Cloud
DRY_RUN = False             # Set True to preview deletions only
# ============================

def delete_temp_test_collections():
    # Connect to Qdrant
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        https=USE_HTTPS,
        api_key=API_KEY,
    )

    print("Connected to Qdrant")

    # Get all collections
    collections = client.get_collections().collections

    if not collections:
        print("No collections found.")
        return

    deleted = []
    skipped = []

    for col in collections:
        name = col.name

        if name.startswith(("temp", "test")):
            if DRY_RUN:
                print(f"[DRY RUN] Would delete: {name}")
            else:
                client.delete_collection(collection_name=name)
                print(f"Deleted collection: {name}")
            deleted.append(name)
        else:
            skipped.append(name)

    print("\n===== SUMMARY =====")
    print(f"Deleted: {len(deleted)} collections")
    print(f"Skipped: {len(skipped)} collections")


if __name__ == "__main__":
    delete_temp_test_collections()