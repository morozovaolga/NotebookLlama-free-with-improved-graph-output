import asyncio
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from notebookllama.processing import parse_file
from notebookllama.workflow import NotebookLMWorkflow, FileInputEvent

TEST_PDF = os.path.join(ROOT, 'data', 'test', 'brain_for_kids.pdf')

async def main():
    if not os.path.exists(TEST_PDF):
        print('Test PDF not found:', TEST_PDF)
        return
    print('Parsing PDF:', TEST_PDF)
    text, images, tables = await parse_file(TEST_PDF)
    if text is None:
        print('Failed to parse PDF')
        return
    print('Parsed text length:', len(text))
    print('First 800 chars:\n', text[:800])

    # Try to run the full workflow if Ollama available
    import urllib.request
    try:
        with urllib.request.urlopen('http://localhost:11434', timeout=2) as resp:
            print('Ollama reachable, running full workflow...')
            wf = NotebookLMWorkflow()
            ev = FileInputEvent(TEST_PDF)
            out = await wf.extract_file_data(ev, type('C',(object,),{'write_event_to_stream': lambda self, ev: None})())
            print('Workflow output summary:\n', out.summary)
            print('Bullet points:', out.highlights)
            print('Mindmap:', out.mind_map)
    except Exception as e:
        print('Ollama not reachable, skipping full workflow:', e)

if __name__ == '__main__':
    asyncio.run(main())
