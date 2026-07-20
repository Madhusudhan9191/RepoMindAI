from backend.app.embedding.embedding_service import EmbeddingService


def main():
    service = EmbeddingService()

    text = """
    def login(username, password):
        return True
    """

    vector = service.embed_text(text)

    print("=" * 60)
    print("Embedding Type :", type(vector))
    print("Embedding Dimension :", len(vector))
    print("First 10 Values :", vector[:10])

    print("=" * 60)

    documents = [
        "def login():",
        "def logout():",
        "class UserService:"
    ]

    vectors = service.embed_documents(documents)

    print("Number of Embeddings :", len(vectors))
    print("Dimension of First Embedding :", len(vectors[0]))

    print("=" * 60)
    print("Model Dimension :", service.get_embedding_dimension())


if __name__ == "__main__":
    main()