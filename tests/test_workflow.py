import pytest
import json

from pydantic import ValidationError
from src.notebookllama.workflow import (
    NotebookLMWorkflow,
    MindMapCreationEvent,
    NotebookOutputEvent,
)
from src.notebookllama.models import Notebook
from workflows import Workflow


@pytest.fixture
def notebook_to_process() -> Notebook:
    return Notebook(
        summary="""The Human Brain:
        The human brain is a complex organ responsible for thought, memory, emotion, and coordination. It contains about 86 billion neurons and operates through electrical and chemical signals. Divided into major parts like the cerebrum, cerebellum, and brainstem, it controls everything from basic survival functions to advanced reasoning. Despite its size, it consumes around 20% of the body’s energy. Neuroscience continues to explore its mysteries, including consciousness and neuroplasticity—its ability to adapt and reorganize.""",
        questions=[
            "How many neurons are in the human brain?",
            "What are the main parts of the human brain?",
            "What percentage of the body's energy does the brain use?",
            "What is neuroplasticity?",
            "What functions is the human brain responsible for?",
        ],
        answers=[
            "About 86 billion neurons.",
            "The cerebrum, cerebellum, and brainstem.",
            "Around 20%.",
            "The brain's ability to adapt and reorganize itself.",
            "Thought, memory, emotion, and coordination.",
        ],
        highlights=[
            "The human brain has about 86 billion neurons.",
            "It controls thought, memory, emotion, and coordination.",
            "Major brain parts include the cerebrum, cerebellum, and brainstem.",
            "The brain uses approximately 20% of the body's energy.",
            "Neuroplasticity allows the brain to adapt and reorganize.",
        ],
    )


def test_init() -> None:
    wf = NotebookLMWorkflow(disable_validation=True)
    assert isinstance(wf, Workflow)
    assert list(wf._get_steps().keys()) == [
        "_done",
        "extract_file_data",
        "generate_mind_map",
    ]


def test_mind_map_event_ser(notebook_to_process: Notebook) -> None:
    json_str = notebook_to_process.model_dump_json()
    json_rep = json.loads(json_str)
    try:
        model = MindMapCreationEvent(
            md_content="Hello",
            **json_rep,
        )
    except ValidationError:
        model = None
    assert isinstance(model, MindMapCreationEvent)
    assert model.md_content == "Hello"
    assert model.model_dump(exclude="md_content") == notebook_to_process.model_dump()
    try:
        model1 = NotebookOutputEvent(
            mind_map="map.html",
            **model.model_dump(
                include={
                    "summary",
                    "highlights",
                    "questions",
                    "answers",
                    "md_content",
                }
            ),
        )
    except ValidationError:
        model1 = None
    assert isinstance(model1, NotebookOutputEvent)
    assert model1.mind_map == "map.html"
    assert model1.model_dump(exclude="mind_map") == model.model_dump()
