import streamlit as st
import requests
import time
import sys
import os

# Add this to prevent recursion issues
sys.setrecursionlimit(10000)

st.set_page_config(page_title="ResearchMate", page_icon="🔬", layout="wide")

# Backend URL - Check if backend is running
BACKEND_URL = "http://localhost:8000"

# Initialize session state with simple types only
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .uploaded-file {
        padding: 5px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 5px 0;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title(" ResearchMate - Document Intelligence")

# Check backend connection
@st.cache_data(ttl=10)
def check_backend_connection():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar for document management
with st.sidebar:
    st.header(" Document Management")
    
    # Backend status
    backend_status = check_backend_connection()
    if backend_status:
        st.success(" Backend connected")
    else:
        st.error(" Backend not connected")
        with st.expander("Troubleshooting"):
            st.markdown("""
            **Start the backend:**
            ```bash
            cd ~/Documents/Master2/Researchmate/backend
            python main.py
            ```
            
            **Then refresh this page.**
            """)
    
    # File upload section
    st.subheader("Upload Document")
    
    # Use a different approach for file uploader to avoid recursion
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        key="file_uploader_widget"  # Fixed key
    )
    
    # Add a separate processing state
    if "upload_in_progress" not in st.session_state:
        st.session_state.upload_in_progress = False
    
    if uploaded_file and not st.session_state.upload_in_progress:
        # Check if file was already uploaded
        already_uploaded = any(
            doc["name"] == uploaded_file.name 
            for doc in st.session_state.uploaded_files
        )
        
        if already_uploaded:
            st.warning(f"⚠️ {uploaded_file.name} was already uploaded.")
        else:
            if st.button(" Upload & Process", type="primary", key="upload_button"):
                st.session_state.upload_in_progress = True
                st.session_state.is_processing = True
                st.rerun()
    
    # Handle upload processing
    if st.session_state.upload_in_progress and uploaded_file:
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            try:
                # Prepare file for upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                
                # Upload to backend
                response = requests.post(
                    f"{BACKEND_URL}/upload", 
                    files=files,
                    timeout=None  # 60 second timeout for upload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "success":
                        # Store simple metadata only
                        st.session_state.uploaded_files.append({
                            "name": uploaded_file.name,
                            "chunks": data.get("chunks_created", 0),
                            "size": data.get("content_length", 0),
                            "id": len(st.session_state.uploaded_files)
                        })
                        st.success(f" {uploaded_file.name} processed successfully!")
                    else:
                        st.error(f" Error: {data.get('message')}")
                else:
                    st.error(f" Server error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Make sure it's running.")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")
            finally:
                # Reset processing state
                time.sleep(1)
                st.session_state.upload_in_progress = False
                st.session_state.is_processing = False
                st.rerun()
    
    # Show uploaded documents
    if st.session_state.uploaded_files:
        st.subheader(" Uploaded Documents")
        for doc in st.session_state.uploaded_files:
            st.markdown(f"""
            <div class="uploaded-file">
                <strong>{doc['name']}</strong><br>
                <small>Chunks: {doc['chunks']} | Size: {doc['size']:,} chars</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Control buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Clear Chat", use_container_width=True, key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button(" Refresh", use_container_width=True, key="refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Stats
    st.markdown("---")
    st.markdown(f"""
    **Stats:**
    - Documents: {len(st.session_state.uploaded_files)}
    - Messages: {len(st.session_state.chat_messages)}
    - Queries: {st.session_state.query_count}
    """)

# Main chat area
st.header(" Chat with Documents")

# Display chat messages
if st.session_state.chat_messages:
    for i, msg in enumerate(st.session_state.chat_messages):
        if i % 2 == 0:  # User message
            with st.chat_message("user"):
                st.write(msg)
        else:  # Assistant message
            with st.chat_message("assistant"):
                st.write(msg)

# Chat input - MUST BE AT MAIN LEVEL, not inside any container
if prompt := st.chat_input("Ask about your documents..."):
    # Don't allow queries while processing or if backend is down
    if st.session_state.is_processing:
        st.warning("Please wait for file upload to complete.")
        st.stop()
    
    if not check_backend_connection():
        st.error("Backend is not available. Please start it first.")
        st.stop()
    
    # Add user message
    st.session_state.chat_messages.append(prompt)
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 Thinking...")
        
        try:
            # Send query to backend
            response = requests.post(
                f"{BACKEND_URL}/query",
                json={
                    "question": prompt,
                    "top_k": 3,
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=120  # 2 minute timeout for query
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "success":
                    answer = data.get("answer", "")
                    
                    # Display answer
                    message_placeholder.empty()
                    st.write(answer)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append(answer)
                    
                    # Increment query count
                    st.session_state.query_count += 1
                    
                    # Show sources if available
                    if data.get("sources"):
                        with st.expander(" Sources", expanded=False):
                            for source in data["sources"]:
                                st.write(f"**{source['filename']}** (score: {source['score']:.3f})")
                else:
                    error_msg = f"Backend error: {data.get('message')}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append(error_msg)
            else:
                error_msg = f"Server error: {response.status_code}"
                st.error(error_msg)
                st.session_state.chat_messages.append(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = "Query timed out. The backend might be processing."
            st.error(error_msg)
            st.session_state.chat_messages.append(error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Lost connection to backend."
            st.error(error_msg)
            st.session_state.chat_messages.append(error_msg)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            st.session_state.chat_messages.append(error_msg)

# Footer
st.markdown("---")
st.caption(f"""
**ResearchMate v1.0** | Backend: {BACKEND_URL} | 
Chat Messages: {len(st.session_state.chat_messages)}
""")

# Debug information (optional)
if st.checkbox("Show Debug Info", value=False):
    with st.expander("Debug Information"):
        st.write("Session State Keys:", list(st.session_state.keys()))
        st.write("Backend Available:", check_backend_connection())
        st.write("Uploaded Files:", st.session_state.uploaded_files)
        
        if st.button("Clear All Cache", key="clear_cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()