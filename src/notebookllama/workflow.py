import json

from workflows import Workflow, step, Context
from workflows.events import StartEvent, StopEvent, Event
from workflows.resource import Resource
from llama_index.tools.mcp import BasicMCPClient
from typing import Annotated, List, Union

MCP_CLIENT = BasicMCPClient(command_or_url="http://localhost:8000/mcp", timeout=120)


class FileInputEvent(StartEvent):
    file: str


class NotebookOutputEvent(StopEvent):
    mind_map: str
    md_content: str
    summary: str
    highlights: List[str]
    questions: List[str]
    answers: List[str]


class MindMapCreationEvent(Event):
    summary: str
    highlights: List[str]
    questions: List[str]
    answers: List[str]
    md_content: str


def get_mcp_client(*args, **kwargs) -> BasicMCPClient:
    return MCP_CLIENT


class NotebookLMWorkflow(Workflow):
    @step
    async def extract_file_data(
        self,
        ev: FileInputEvent,
        mcp_client: Annotated[BasicMCPClient, Resource(get_mcp_client)],
        ctx: Context,
    ) -> Union[MindMapCreationEvent, NotebookOutputEvent]:
        ctx.write_event_to_stream(ev=ev)
        result = await mcp_client.call_tool(
            tool_name="process_file_tool", arguments={"filename": ev.file}
        )
        split_result = result.content[0].text.split("\n%separator%\n")
        json_data = split_result[0]
        md_text = split_result[1] if len(split_result) > 1 else ""
        # Проверка на ошибку или невалидный JSON
        if json_data == "Sorry, your file could not be processed.":
            return NotebookOutputEvent(
                mind_map="Unprocessable file, sorry😭",
                md_content="",
                summary="",
                highlights=[],
                questions=[],
                answers=[],
            )
        try:
            json_rep = json.loads(json_data)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_text = f"Ошибка при обработке файла: {json_data}\nТип ошибки: {type(e).__name__}\n{str(e)}\nТрейсбек:\n{tb}"
            return NotebookOutputEvent(
                mind_map=error_text,
                md_content="",
                summary="",
                highlights=[],
                questions=[],
                answers=[],
            )
        return MindMapCreationEvent(
            md_content=md_text,
            **json_rep,
        )

    @step
    async def generate_mind_map(
        self,
        ev: MindMapCreationEvent,
        mcp_client: Annotated[BasicMCPClient, Resource(get_mcp_client)],
        ctx: Context,
    ) -> NotebookOutputEvent:
        ctx.write_event_to_stream(
            ev=ev,
        )
        result = await mcp_client.call_tool(
            tool_name="get_mind_map_tool",
            arguments={"summary": ev.summary, "highlights": ev.highlights},
        )
        if result is not None:
            return NotebookOutputEvent(
                mind_map=result.content[0].text,
                **ev.model_dump(
                    include={
                        "summary",
                        "highlights",
                        "questions",
                        "answers",
                        "md_content",
                    }
                ),
            )
        error_msg = getattr(result, 'error', None)
        if error_msg:
            mind_map_text = f"Не удалось построить майндмэп: {error_msg}"
        else:
            mind_map_text = "Не удалось построить майндмэп. Проверьте настройки модели и ключа API."
        return NotebookOutputEvent(
            mind_map=mind_map_text,
            **ev.model_dump(
                include={
                    "summary",
                    "highlights",
                    "questions",
                    "answers",
                    "md_content",
                }
            ),
        )
