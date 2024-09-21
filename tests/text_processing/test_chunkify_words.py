from text_generation.text_processing import chunkify_words


def test_chunkify_words_chunks_simple_sentence_correctly():
    """The function should chunkify a simple sentence correctly"""
    words = ["hello", "world", "this", "is", "a", "test"]
    chunks = chunkify_words(words, min_length=1, max_length=20, target_length_decider=lambda: 11)
    word_chunks = tuple(chunk.text for chunk in chunks)

    assert word_chunks == ("hello world", "this is a", "test")


def test_chunkify_words_chunks_simple_sentence_correctly_with_min_length():
    """The last chunk should be discarded if it is too short"""
    words = ["hello", "world", "a"]
    chunks = chunkify_words(words, min_length=2, max_length=20, target_length_decider=lambda: 11)
    word_chunks = tuple(chunk.text for chunk in chunks)

    assert word_chunks == ("hello world",)


def test_current_chunk_to_short_but_candidate_word_to_long():
    words = ["Hi", "thisisasuperlongword"]
    chunks = chunkify_words(words, min_length=1, max_length=20, target_length_decider=lambda: 10)
    word_chunks = tuple(chunk.text for chunk in chunks)

    assert word_chunks == ("Hi", "thisisasuperlongword")
