import pytest
import os
import socket
from dotenv import load_dotenv
from typing import List

from src.notebookllama.documents import DocumentManager, ManagedDocument
from sqlalchemy import text, Table

ENV = load_dotenv()


def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is open on a given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        return result == 0


@pytest.fixture
def documents() -> List[ManagedDocument]:
    return [
        ManagedDocument(
            document_name="Project Plan",
            content="This is the full content of the project plan document.",
            summary="A summary of the project plan.",
            q_and_a="Q: What is the goal? A: To deliver the project.",
            mindmap="Project -> Tasks -> Timeline",
            bullet_points="• Define scope\n• Assign tasks\n• Set deadlines",
        ),
        ManagedDocument(
            document_name="Meeting Notes",
            content="Notes from the weekly team meeting.",
            summary="Summary of meeting discussions.",
            q_and_a="Q: Who attended? A: All team members.",
            mindmap="Meeting -> Topics -> Decisions",
            bullet_points="• Discussed progress\n• Identified blockers\n• Planned next steps",
        ),
        ManagedDocument(
            document_name="Research Article",
            content="Content of the research article goes here.",
            summary="Key findings from the research.",
            q_and_a="Q: What was discovered? A: New insights into the topic.",
            mindmap="Research -> Methods -> Results",
            bullet_points="• Literature review\n• Data analysis\n• Conclusions",
        ),
        ManagedDocument(
            document_name="User Guide",
            content="Instructions for using the application.",
            summary="Overview of user guide contents.",
            q_and_a="Q: How to start? A: Follow the setup instructions.",
            mindmap="Guide -> Sections -> Steps",
            bullet_points="• Installation\n• Configuration\n• Usage tips",
        ),
    ]


@pytest.mark.skipif(
    condition=not is_port_open(host="localhost", port=5432) and not ENV,
    reason="Either Postgres is currently unavailable or you did not set any env variables in a .env file",
)
def test_document_manager(documents: List[ManagedDocument]) -> None:
    engine_url = f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}"
    manager = DocumentManager(engine_url=engine_url, table_name="test_documents")
    assert not manager.table
    manager.connection.execute(text("DROP TABLE IF EXISTS test_documents;"))
    manager.connection.commit()
    manager._create_table()
    assert isinstance(manager.table, Table)
    manager.put_documents(documents=documents)
    names = manager.get_names()
    assert names == [doc.document_name for doc in documents]
    docs = manager.get_documents()
    assert docs == documents
    docs1 = manager.get_documents(names=["Project Plan", "Meeting Notes"])
    assert len(docs1) == 2
