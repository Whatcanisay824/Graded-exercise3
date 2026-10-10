from rag import retrieve_documents

def get_context(query, top_k=5):
    docs = retrieve_documents(query, top_k)
    if not docs:
        return "No relevant information found in the knowledge base."
    parts = []
    for d in docs:
        policy_id = d.metadata.get("policy_id", "unknown")
        parts.append(f"[{policy_id}] {d.page_content}")
    return "\n\n".join(parts)