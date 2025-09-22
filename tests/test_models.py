import pytest

from src.notebookllama.models import (
    Notebook,
)
from src.notebookllama.verifying import ClaimVerification
from src.notebookllama.mindmap import MindMap, Node, Edge
from src.notebookllama.audio import MultiTurnConversation, ConversationTurn
from src.notebookllama.documents import ManagedDocument
from src.notebookllama.audio import (
    PodcastConfig,
    VoiceConfig,
    AudioQuality,
)
from pydantic import ValidationError


def test_notebook() -> None:
    n1 = Notebook(
        summary="This is a summary",
        questions=[
            "What is the capital of Spain?",
            "What is the capital of France?",
            "What is the capital of Italy?",
            "What is the capital of Portugal?",
            "What is the capital of Germany?",
        ],
        answers=[
            "Madrid",
            "Paris",
            "Rome",
            "Lisbon",
            "Berlin",
        ],
        highlights=["This", "is", "a", "summary"],
    )
    assert n1.summary == "This is a summary"
    assert n1.questions[0] == "What is the capital of Spain?"
    assert n1.answers[0] == "Madrid"
    assert n1.highlights[0] == "This"
    # Fewer answers than questions
    with pytest.raises(ValidationError):
        Notebook(
            summary="This is a summary",
            questions=[
                "What is the capital of France?",
                "What is the capital of Italy?",
                "What is the capital of Portugal?",
                "What is the capital of Germany?",
            ],
            answers=[
                "Paris",
                "Rome",
                "Lisbon",
            ],
            highlights=["This", "is", "a", "summary"],
        )
    # Fewer highlights than required
    with pytest.raises(ValidationError):
        Notebook(
            summary="This is a summary",
            questions=[
                "What is the capital of Spain?",
                "What is the capital of France?",
                "What is the capital of Italy?",
                "What is the capital of Portugal?",
                "What is the capital of Germany?",
            ],
            answers=[
                "Madrid",
                "Paris",
                "Rome",
                "Lisbon",
                "Berlin",
            ],
            highlights=["This", "is"],
        )


def test_mind_map() -> None:
    m1 = MindMap(
        nodes=[
            Node(id="A", content="Auxin is released"),
            Node(id="B", content="Travels to the roots"),
            Node(id="C", content="Root cells grow"),
        ],
        edges=[
            Edge(from_id="A", to_id="B"),
            Edge(from_id="A", to_id="C"),
            Edge(from_id="B", to_id="C"),
        ],
    )
    assert m1.nodes[0].id == "A"
    assert m1.nodes[0].content == "Auxin is released"
    assert m1.edges[0].from_id == "A"
    assert m1.edges[0].to_id == "B"

    with pytest.raises(ValidationError):
        MindMap(
            nodes=[
                Node(id="A", content="Auxin is released"),
                Node(id="B", content="Travels to the roots"),
                Node(id="C", content="Root cells grow"),
            ],
            edges=[
                Edge(from_id="A", to_id="B"),
                Edge(from_id="A", to_id="D"),  # "D" does not exist
                Edge(from_id="B", to_id="C"),
            ],
        )


def test_multi_turn_conversation() -> None:
    turns = [
        ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
        ConversationTurn(speaker="speaker2", content="I am very well, how about you?"),
        ConversationTurn(speaker="speaker1", content="I am well too, thanks!"),
    ]
    assert turns[0].speaker == "speaker1"
    assert turns[0].content == "Hello, who are you?"
    conversation = MultiTurnConversation(
        conversation=turns,
    )
    assert isinstance(conversation.conversation, list)
    assert isinstance(conversation.conversation[0], ConversationTurn)
    wrong_turns = [
        ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
        ConversationTurn(speaker="speaker2", content="I am very well, how about you?"),
    ]
    wrong_turns1 = [
        ConversationTurn(speaker="speaker2", content="Hello, who are you?"),
        ConversationTurn(speaker="speaker1", content="I am very well, how about you?"),
        ConversationTurn(speaker="speaker2", content="I am well too!"),
    ]
    wrong_turns2 = [
        ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
        ConversationTurn(speaker="speaker1", content="How is your life going?"),
        ConversationTurn(
            speaker="speaker2",
            content="What is all this interest in me all of a sudden?!",
        ),
    ]
    wrong_turns3 = [
        ConversationTurn(speaker="speaker1", content="Hello, who are you?"),
        ConversationTurn(speaker="speaker2", content="I'm well! But..."),
        ConversationTurn(
            speaker="speaker2",
            content="...What is all this interest in me all of a sudden?!",
        ),
    ]
    with pytest.raises(ValidationError):
        MultiTurnConversation(conversation=wrong_turns)
    with pytest.raises(ValidationError):
        MultiTurnConversation(conversation=wrong_turns1)
    with pytest.raises(ValidationError):
        MultiTurnConversation(conversation=wrong_turns2)
    with pytest.raises(ValidationError):
        MultiTurnConversation(conversation=wrong_turns3)


def test_claim_verification() -> None:
    cl1 = ClaimVerification(
        claim_is_true=True, supporting_citations=["Support 1", "Support 2"]
    )
    assert cl1.claim_is_true
    assert cl1.supporting_citations == ["Support 1", "Support 2"]
    cl2 = ClaimVerification(
        claim_is_true=False, supporting_citations=["Support 1", "Support 2"]
    )
    assert cl2.supporting_citations == ["The claim was deemed false."]
    cl3 = ClaimVerification(
        claim_is_true=False,
    )
    assert cl3.supporting_citations is None
    with pytest.raises(ValidationError):
        ClaimVerification(
            claim_is_true=True,
            supporting_citations=["Support 1", "Support 2", "Support 3", "Support 4"],
        )


def test_managed_documents() -> None:
    d1 = ManagedDocument(
        document_name="Hello World",
        content="This is a test",
        summary="Test",
        q_and_a="Hello? World.",
        mindmap="Hello -> World",
        bullet_points=". Hello, . World",
    )
    assert d1.document_name == "Hello World"
    assert d1.content == "This is a test"
    assert d1.summary == "Test"
    assert d1.q_and_a == "Hello? World."
    assert d1.mindmap == "Hello -> World"
    assert d1.bullet_points == ". Hello, . World"
    d2 = ManagedDocument(
        document_name="Hello World",
        content="This is a test",
        summary="Test's child",
        q_and_a="Hello? World.",
        mindmap="Hello -> World",
        bullet_points=". Hello, . World",
    )
    assert d2.summary == "Test's child"


# Test Audio Configuration Models
def test_voice_config_defaults():
    """Test VoiceConfig default values"""
    config = VoiceConfig()
    assert config.speaker1_voice_id == "nPczCjzI2devNBz1zQrb"
    assert config.speaker2_voice_id == "Xb7hH8MSUJpSbSDYk0k2"
    assert config.model_id == "eleven_turbo_v2_5"
    assert config.output_format == "mp3_22050_32"


def test_voice_config_custom_values():
    """Test VoiceConfig with custom values"""
    config = VoiceConfig(
        speaker1_voice_id="custom_voice_1",
        speaker2_voice_id="custom_voice_2",
        model_id="custom_model",
        output_format="wav_44100_16",
    )
    assert config.speaker1_voice_id == "custom_voice_1"
    assert config.speaker2_voice_id == "custom_voice_2"
    assert config.model_id == "custom_model"
    assert config.output_format == "wav_44100_16"


def test_audio_quality_defaults():
    """Test AudioQuality default values"""
    config = AudioQuality()
    assert config.bitrate == "320k"
    assert config.quality_params == ["-q:a", "0"]


def test_audio_quality_custom_values():
    """Test AudioQuality with custom values"""
    custom_params = ["-q:a", "2", "-compression_level", "5"]
    config = AudioQuality(bitrate="256k", quality_params=custom_params)
    assert config.bitrate == "256k"
    assert config.quality_params == custom_params


def test_podcast_config_defaults():
    """Test that PodcastConfig creates with proper defaults"""
    config = PodcastConfig()

    assert config.style == "conversational"
    assert config.tone == "friendly"
    assert config.focus_topics is None
    assert config.target_audience == "general"
    assert config.custom_prompt is None
    assert config.speaker1_role == "host"
    assert config.speaker2_role == "guest"
    assert isinstance(config.voice_config, VoiceConfig)
    assert isinstance(config.audio_quality, AudioQuality)


def test_podcast_config_custom_values():
    """Test that PodcastConfig accepts custom values"""
    focus_topics = ["AI Ethics", "Machine Learning", "Future Tech"]
    custom_prompt = "Make it engaging and technical"

    config = PodcastConfig(
        style="interview",
        tone="professional",
        focus_topics=focus_topics,
        target_audience="expert",
        custom_prompt=custom_prompt,
        speaker1_role="interviewer",
        speaker2_role="technical_expert",
    )

    assert config.style == "interview"
    assert config.tone == "professional"
    assert config.focus_topics == focus_topics
    assert config.target_audience == "expert"
    assert config.custom_prompt == custom_prompt
    assert config.speaker1_role == "interviewer"
    assert config.speaker2_role == "technical_expert"


def test_podcast_config_validation():
    """Test that PodcastConfig validates input values"""
    # Test invalid style
    with pytest.raises(ValidationError):
        PodcastConfig(style="invalid_style")

    # Test invalid tone
    with pytest.raises(ValidationError):
        PodcastConfig(tone="invalid_tone")

    # Test invalid target_audience
    with pytest.raises(ValidationError):
        PodcastConfig(target_audience="invalid_audience")


def test_conversation_turn():
    """Test ConversationTurn model"""
    turn = ConversationTurn(speaker="speaker1", content="Hello world")
    assert turn.speaker == "speaker1"
    assert turn.content == "Hello world"
