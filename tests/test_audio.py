import pytest

from elevenlabs import AsyncElevenLabs
from src.notebookllama.audio import (
    PodcastGenerator,
    MultiTurnConversation,
    PodcastConfig,
    AudioGenerationError,
    ConversationGenerationError,
    PodcastGeneratorError,
)
from llama_index.core.llms.structured_llm import StructuredLLM
from llama_index.core.llms import MockLLM
from pydantic import BaseModel, ValidationError


class MockElevenLabs(AsyncElevenLabs):
    def __init__(self, test_api_key: str) -> None:
        self.test_api_key = test_api_key


class DataModel(BaseModel):
    test: str


@pytest.fixture()
def correct_structured_llm() -> StructuredLLM:
    return MockLLM().as_structured_llm(MultiTurnConversation)


@pytest.fixture()
def wrong_structured_llm() -> StructuredLLM:
    return MockLLM().as_structured_llm(DataModel)


def test_podcast_generator_model(
    correct_structured_llm: StructuredLLM, wrong_structured_llm: StructuredLLM
) -> None:
    n = PodcastGenerator(
        client=MockElevenLabs(test_api_key="a"), llm=correct_structured_llm
    )
    assert isinstance(n.client, AsyncElevenLabs)
    assert isinstance(n.llm, StructuredLLM)
    assert n.llm.output_cls == MultiTurnConversation
    with pytest.raises(ValidationError):
        PodcastGenerator(
            client=MockElevenLabs(test_api_key="a"), llm=wrong_structured_llm
        )


# Test Error Classes
def test_podcast_generator_error_hierarchy():
    """Test custom exception hierarchy"""
    assert issubclass(AudioGenerationError, PodcastGeneratorError)
    assert issubclass(ConversationGenerationError, PodcastGeneratorError)
    assert issubclass(PodcastGeneratorError, Exception)


def test_custom_exceptions():
    """Test custom exception instantiation"""
    base_error = PodcastGeneratorError("Base error")
    assert str(base_error) == "Base error"

    audio_error = AudioGenerationError("Audio error")
    assert str(audio_error) == "Audio error"
    assert isinstance(audio_error, PodcastGeneratorError)

    conversation_error = ConversationGenerationError("Conversation error")
    assert str(conversation_error) == "Conversation error"
    assert isinstance(conversation_error, PodcastGeneratorError)


def test_build_conversation_prompt_basic(correct_structured_llm: StructuredLLM):
    """Test basic prompt building functionality"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    config = PodcastConfig()
    transcript = "This is a test transcript about artificial intelligence."

    prompt = generator._build_conversation_prompt(transcript, config)

    assert "conversational podcast conversation" in prompt
    assert "CONVERSATION STYLE: conversational" in prompt
    assert "TONE: friendly" in prompt
    assert "TARGET AUDIENCE: general" in prompt
    assert "Speaker 1: host" in prompt
    assert "Speaker 2: guest" in prompt
    assert transcript in prompt
    assert "Balance accessibility with depth" in prompt


def test_build_conversation_prompt_with_focus_topics(
    correct_structured_llm: StructuredLLM,
):
    """Test prompt building with focus topics"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    focus_topics = ["Machine Learning", "Ethics in AI", "Future Applications"]
    config = PodcastConfig(focus_topics=focus_topics)
    transcript = "AI transcript content"

    prompt = generator._build_conversation_prompt(transcript, config)

    assert "FOCUS TOPICS: Make sure to discuss these topics in detail:" in prompt
    for topic in focus_topics:
        assert f"- {topic}" in prompt


def test_build_conversation_prompt_technical_audience(
    correct_structured_llm: StructuredLLM,
):
    """Test prompt building for technical audience"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    config = PodcastConfig(target_audience="technical", style="educational")
    transcript = "Technical content about neural networks"

    prompt = generator._build_conversation_prompt(transcript, config)

    assert "educational podcast conversation" in prompt
    assert "TARGET AUDIENCE: technical" in prompt
    assert "Use technical terminology appropriately" in prompt


def test_build_conversation_prompt_beginner_audience(
    correct_structured_llm: StructuredLLM,
):
    """Test prompt building for beginner audience"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    config = PodcastConfig(target_audience="beginner", tone="casual")
    transcript = "Complex topic simplified"

    prompt = generator._build_conversation_prompt(transcript, config)

    assert "TONE: casual" in prompt
    assert "TARGET AUDIENCE: beginner" in prompt
    assert "Explain concepts clearly and avoid jargon" in prompt


def test_build_conversation_prompt_with_custom_prompt(
    correct_structured_llm: StructuredLLM,
):
    """Test prompt building with custom instructions"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    custom_prompt = (
        "Make sure to include practical examples and keep it under 10 minutes"
    )
    config = PodcastConfig(custom_prompt=custom_prompt)
    transcript = "Sample content"

    prompt = generator._build_conversation_prompt(transcript, config)

    assert f"ADDITIONAL INSTRUCTIONS: {custom_prompt}" in prompt


def test_build_conversation_prompt_custom_speaker_roles(
    correct_structured_llm: StructuredLLM,
):
    """Test prompt building with custom speaker roles"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    config = PodcastConfig(
        speaker1_role="professor", speaker2_role="student", style="interview"
    )
    transcript = "Educational content"

    prompt = generator._build_conversation_prompt(transcript, config)

    assert "Speaker 1: professor" in prompt
    assert "Speaker 2: student" in prompt
    assert "interview podcast conversation" in prompt


def test_build_conversation_prompt_all_audiences(correct_structured_llm: StructuredLLM):
    """Test that all audience types have appropriate instructions"""
    generator = PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )

    audience_expectations = {
        "technical": "Use technical terminology appropriately",
        "beginner": "Explain concepts clearly and avoid jargon",
        "expert": "Assume advanced knowledge and discuss nuanced aspects",
        "business": "Focus on practical applications, ROI, and strategic implications",
        "general": "Balance accessibility with depth",
    }

    for audience, expected_instruction in audience_expectations.items():
        config = PodcastConfig(target_audience=audience)
        prompt = generator._build_conversation_prompt("test content", config)
        assert expected_instruction in prompt


@pytest.fixture()
def sample_podcast_generator(correct_structured_llm: StructuredLLM) -> PodcastGenerator:
    """Fixture providing a configured PodcastGenerator for testing"""
    return PodcastGenerator(
        client=MockElevenLabs(test_api_key="test"), llm=correct_structured_llm
    )
