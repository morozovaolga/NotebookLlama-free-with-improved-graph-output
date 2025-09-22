import os
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from typing import List, Optional

from documents import DocumentManager, ManagedDocument

# Load environment variables
load_dotenv()

# Initialize the document manager
engine_url = f"postgresql+psycopg2://{os.getenv('pgql_user')}:{os.getenv('pgql_psw')}@localhost:5432/{os.getenv('pgql_db')}"
document_manager = DocumentManager(engine_url=engine_url)


def fetch_documents(names: Optional[List[str]]) -> List[ManagedDocument]:
    """Retrieve documents from the database"""
    return document_manager.get_documents(names=names)


def fetch_document_names() -> List[str]:
    return document_manager.get_names()


def display_document(document: ManagedDocument) -> None:
    """Display a single document in an expandable format"""
    with st.expander(f"📄 {document.document_name}"):
        # Summary section
        st.markdown("## Summary")
        st.markdown(document.summary)

        # Bullet Points section
        st.markdown(document.bullet_points)

        # FAQ section (nested expander)
        with st.expander("FAQ"):
            st.markdown(document.q_and_a)

        # Mind Map section
        if document.mindmap:
            st.markdown("## Mind Map")
            components.html(document.mindmap, height=800, scrolling=True)


def main():
    # Display the network
    st.set_page_config(
        page_title="NotebookLlaMa - Document Management",
        page_icon="📚",
        layout="wide",
        menu_items={
            "Get Help": "https://github.com/run-llama/notebooklm-clone/discussions/categories/general",
            "Report a bug": "https://github.com/run-llama/notebooklm-clone/issues/",
            "About": "An OSS alternative to NotebookLM that runs with the power of a flully Llama!",
        },
    )
    st.sidebar.header("Document Management📚")
    st.sidebar.info("To switch to the other pages, select it from above!🔺")
    st.markdown("---")
    st.markdown("## NotebookLlaMa - Document Management📚")

    # Slider for number of documents
    names = st.multiselect(
        options=fetch_document_names(),
        default=None,
        label="Select the Documents you want to display",
    )

    # Button to load documents
    if st.button("Load Documents", type="primary"):
        with st.spinner("Loading documents..."):
            try:
                documents = fetch_documents(names)

                if documents:
                    st.success(f"Successfully loaded {len(documents)} document(s)")
                    st.session_state.documents = documents
                else:
                    st.warning("No documents found in the database.")
                    st.session_state.documents = []

            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                st.session_state.documents = []

    # Display documents if they exist in session state
    if "documents" in st.session_state and st.session_state.documents:
        st.markdown("## Documents")

        # Display each document
        for i, document in enumerate(st.session_state.documents):
            display_document(document)

            # Add some spacing between documents
            if i < len(st.session_state.documents) - 1:
                st.markdown("---")

    elif "documents" in st.session_state:
        st.info(
            "No documents to display. Try adjusting the limit and clicking 'Load Documents'."
        )

    else:
        st.info("Click 'Load Documents' to view your processed documents.")


if __name__ == "__main__":
    main()
