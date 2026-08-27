from services.ragservice import RagService


def test_build_context_joins_page_content():
    service = RagService()
    docs = [
        type("Doc", (), {"page_content": "First chunk"})(),
        type("Doc", (), {"page_content": "Second chunk"})(),
    ]

    context = service.build_context(docs)

    assert context == "First chunk\n\nSecond chunk"


def test_chat_history_is_shared_by_session_in_langchain_in_memory_store():
    service = RagService()
    session_id = "test-in-memory-session"
    service.session_store.mdelete([service._session_key(session_id)])

    service.append_history(session_id, "What was uploaded?", "A PDF document.")

    another_service = RagService()
    assert another_service.build_history(session_id) == (
        "User: What was uploaded?\nAssistant: A PDF document."
    )
    stored_session = another_service.session_store.mget([
        another_service._session_key(session_id)
    ])[0]
    assert stored_session["history_by_upload"]["__no_upload__"].messages[0].content == "What was uploaded?"


def test_new_upload_uses_a_separate_chat_history():
    service = RagService()
    session_id = "test-history-by-upload"
    service.session_store.mdelete([service._session_key(session_id)])

    service.set_session_upload_id(session_id, "first-upload")
    service.append_history(session_id, "What is in the first PDF?", "First document content.")
    service.set_session_upload_id(session_id, "second-upload")

    assert service.build_history(session_id) == ""

    service.append_history(session_id, "What is in the second PDF?", "Second document content.")
    service.set_session_upload_id(session_id, "first-upload")
    assert service.build_history(session_id) == (
        "User: What is in the first PDF?\nAssistant: First document content."
    )
