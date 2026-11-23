from voice_rag.conversation import ConversationQuery, SYSTEM_PROMPT


def test_followup_expands_query_from_previous_utterance():
    state = ConversationQuery()
    first, retrieve1 = state.prepare("شرایط عضویت ویژه چیست؟")
    second, retrieve2 = state.prepare("هزینه اون چنده؟")
    assert first == "شرایط عضویت ویژه چیست؟"
    assert retrieve1 and retrieve2
    assert second.startswith(first)


def test_long_new_topic_does_not_inherit_older_query():
    state = ConversationQuery()
    state.prepare("ساعت کار کتابخانه چنده؟")
    new, _ = state.prepare("درباره روش تمدید کتاب در روزهای تعطیل توضیح بده")
    assert new == "درباره روش تمدید کتاب در روزهای تعطیل توضیح بده"


def test_social_chat_skips_retrieval():
    state = ConversationQuery()
    assert state.prepare("سلام")[1] is False
    assert state.prepare("ممنون")[1] is False
    assert "مکالمه" in SYSTEM_PROMPT


def test_empty_utterance():
    assert ConversationQuery().prepare("  ") == ("", False)


def test_social_chat_does_not_replace_last_topic():
    state = ConversationQuery()
    state.prepare("در مورد عضویت ویژه بگو")
    state.prepare("ممنون")
    query, _ = state.prepare("هزینه اون چنده؟")
    assert query.startswith("در مورد عضویت ویژه بگو")
