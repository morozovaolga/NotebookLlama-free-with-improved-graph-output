import streamlit as st
import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifying import verify_claim as sync_verify_claim
from llama_index.tools.mcp import BasicMCPClient

MCP_CLIENT = BasicMCPClient(command_or_url="http://localhost:8000/mcp")


async def chat(inpt: str):
    result = await MCP_CLIENT.call_tool(
        tool_name="query_index_tool", arguments={"question": inpt}
    )
    return result.content[0].text


def sync_chat(inpt: str):
    return asyncio.run(chat(inpt=inpt))


# Chat Interface
st.set_page_config(page_title="NotebookLlaMa - Document Chat", page_icon="🗣")

st.sidebar.header("Document Chat🗣")
st.sidebar.info("To switch to the Home page, select it from above!🔺")
st.markdown("---")
st.markdown("## NotebookLlaMa - Document Chat🗣")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sources" in message:
            # Display the main response
            st.markdown(message["content"])
            # Add toggle for sources
            with st.expander("Sources"):
                st.markdown(message["sources"])
        elif message["role"] == "assistant" and "verification" in message:
            # Display verification results
            st.markdown(message["content"])
            verification = message["verification"]

            # Show verification status with appropriate styling
            if verification["is_true"]:
                st.success("✅ Claim verified as TRUE")
            else:
                st.error("❌ Claim verified as FALSE")

            # Show citations if available
            if verification["citations"]:
                with st.expander("Supporting Citations"):
                    for i, citation in enumerate(verification["citations"], 1):
                        st.markdown(f"**Citation {i}:**")
                        st.markdown(f"*{citation}*")
                        st.markdown("---")
            else:
                st.info("No supporting citations found")
        else:
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your document"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = sync_chat(prompt)

                # Split response and sources if they exist
                # Assuming your response format includes sources somehow
                # You might need to modify this based on your actual response format
                if "## Sources" in response:
                    parts = response.split("## Sources", 1)
                    main_response = parts[0].strip()
                    sources = "## Sources" + parts[1].strip()
                else:
                    main_response = response
                    sources = None

                st.markdown(main_response)

                # Add toggle for sources if they exist
                if sources:
                    with st.expander("Sources"):
                        st.markdown(sources)
                    # Add to history with sources
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": main_response,
                            "sources": sources,
                        }
                    )
                else:
                    # Add to history without sources
                    st.session_state.messages.append(
                        {"role": "assistant", "content": main_response}
                    )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# Claim Verification Section
st.markdown("---")
st.markdown("### 🔍 Claim Verification")

# Check if there are any assistant messages with sources to verify
assistant_messages_with_sources = [
    msg
    for msg in st.session_state.messages
    if msg["role"] == "assistant" and "sources" in msg
]

if assistant_messages_with_sources:
    st.markdown("Select a response to verify its claims against the sources:")

    # Create a selectbox with assistant responses
    response_options = []
    for i, msg in enumerate(assistant_messages_with_sources):
        # Truncate long responses for display
        content_preview = (
            msg["content"][:100] + "..."
            if len(msg["content"]) > 100
            else msg["content"]
        )
        response_options.append(f"Response {i + 1}: {content_preview}")

    selected_response_idx = st.selectbox(
        "Choose response to verify:",
        range(len(response_options)),
        format_func=lambda x: response_options[x],
    )

    # Create columns for the verify button
    col1, col2 = st.columns([4, 1])

    with col2:
        verify_button = st.button("🔍 Verify Claims", type="primary")

    # Handle claim verification
    if verify_button:
        selected_message = assistant_messages_with_sources[selected_response_idx]
        claim_text = selected_message["content"]
        sources_text = selected_message["sources"]

        with st.spinner("Verifying claims against sources..."):
            try:
                # Call the verify_claim function with claim and sources
                is_true, citations = sync_verify_claim(claim_text, sources_text)

                # Create verification result message
                verification_result = {"is_true": is_true, "citations": citations}

                # Display verification result
                st.markdown("### Verification Result:")

                # Show the claim being verified (truncated for display)
                claim_preview = (
                    claim_text[:200] + "..." if len(claim_text) > 200 else claim_text
                )
                st.markdown(f"**Claim:** {claim_preview}")

                # Show verification status with appropriate styling
                if is_true:
                    st.success("✅ Claims verified as TRUE")
                else:
                    st.error("❌ Claims verified as FALSE")

                # Show citations if available
                if citations:
                    with st.expander("Supporting Citations", expanded=True):
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"**Citation {i}:**")
                            st.markdown(f"*{citation}*")
                            st.markdown("---")
                else:
                    st.info("No supporting citations found")

                # Add verification to chat history
                verification_message = f"**Claim Verification:** {claim_preview}\n\n"
                verification_message += "**Result:** " + (
                    "✅ TRUE" if is_true else "❌ FALSE"
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": verification_message,
                        "verification": verification_result,
                    }
                )

            except Exception as e:
                error_msg = f"Error verifying claims: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

else:
    st.info(
        "No assistant responses with sources available to verify. Chat with the document first to generate responses with sources."
    )

# Optional: Add some helpful information
with st.expander("ℹ️ About Claim Verification"):
    st.markdown("""
    **How it works:**
    - Select an assistant response that includes sources
    - The system will verify the claims made in that response against the provided sources
    - You'll receive a TRUE/FALSE result with supporting citations if available

    **What gets verified:**
    - The assistant's response content is treated as the claim
    - The verification is done against the sources that were provided with that response
    - This helps ensure the assistant's answers are backed by the actual document content

    **Note:**
    - Only responses that include sources can be verified
    - The verification checks if the assistant's claims are supported by the cited sources
    """)
