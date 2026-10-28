import streamlit as st
from pathlib import Path

from src.pipeline import MultimodalRAG
from index_pdf import index_pdf

from src.similarity.plagiarism_checker import (
    PlagiarismChecker
)

from src.similarity.web_search import (
    WebPlagiarismSearch
)

from src.similarity.pdf_highlighter import (
    PDFHighlighter
)

from src.similarity.ai_detector import (
    AIDetector
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multimodal RAG + AI Plagiarism Checker",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = Path(
    "data/uploads"
)

OUTPUT_DIR = Path(
    "storage/similarity"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "document_path" not in st.session_state:
    st.session_state.document_path = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "plagiarism_result" not in st.session_state:
    st.session_state.plagiarism_result = None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.7;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.3);
        line-height: 1.7;
        margin-bottom: 20px;
    }

    .source-box {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📚 Multimodal RAG + 🔎 AI Plagiarism Checker'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload one PDF, ask questions, detect online text similarity, '
    'and analyze AI-like writing locally.'
    '</div>',
    unsafe_allow_html=True
)


st.divider()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "## 📄 Upload Document"
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help=(
        "Upload a research paper, report, "
        "or any other PDF."
    ),
    key="pdf_uploader"
)


# ============================================================
# NO FILE
# ============================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a PDF above to get started."
    )

    st.stop()


# ============================================================
# SAVE PDF
# ============================================================

pdf_path = (
    UPLOAD_DIR
    / uploaded_file.name
)


with open(
    pdf_path,
    "wb"
) as f:

    f.write(
        uploaded_file.getbuffer()
    )


# ============================================================
# DETECT NEW DOCUMENT
# ============================================================

if (
    st.session_state.document_name
    != uploaded_file.name
):

    st.session_state.rag = None

    st.session_state.document_name = (
        uploaded_file.name
    )

    st.session_state.document_path = (
        str(pdf_path)
    )

    st.session_state.indexed = False

    st.session_state.last_result = None

    st.session_state.plagiarism_result = None


st.success(
    f"Uploaded: **{uploaded_file.name}**"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def prepare_document(
    file_path
):

    file_path = Path(
        file_path
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )


    # --------------------------------------------------------
    # Use the EXISTING indexing function.
    #
    # This is the same function that your original
    # working RAG application used.
    # --------------------------------------------------------

    stats = index_pdf(
        file_path,
        reset=True
    )


    # --------------------------------------------------------
    # Create the existing RAG object.
    # --------------------------------------------------------

    rag = MultimodalRAG(
        top_k=5
    )


    return rag, stats


# ============================================================
# ============================================================
# RAG SECTION
# ============================================================
# ============================================================

st.divider()

st.markdown(
    "## 💬 Document Question Answering"
)

st.caption(
    "Uses your existing multimodal RAG pipeline."
)


# ============================================================
# PREPARE BUTTON
# ============================================================

prepare_button = st.button(
    "📚 Prepare Document for RAG",
    use_container_width=True,
    key="prepare_document_button"
)


if prepare_button:

    with st.status(
        "Preparing document...",
        expanded=True
    ) as status:

        try:

            st.write(
                "📄 Extracting PDF content..."
            )


            rag, stats = prepare_document(
                pdf_path
            )


            st.write(
                "🧠 Loading retrieval system..."
            )


            st.session_state.rag = rag

            st.session_state.document_path = (
                str(pdf_path)
            )

            st.session_state.document_name = (
                uploaded_file.name
            )

            st.session_state.indexed = True

            st.session_state.last_result = None


            status.update(
                label="Document prepared successfully!",
                state="complete"
            )


            st.success(
                f"""
                **{uploaded_file.name}** is ready.

                • Text chunks: {stats.get("text_chunks", 0)}

                • Tables: {stats.get("tables", 0)}

                • Page images: {stats.get("page_images", 0)}
                """
            )


        except Exception as e:

            st.session_state.rag = None

            st.session_state.indexed = False


            status.update(
                label="RAG indexing failed",
                state="error"
            )


            st.error(
                f"{type(e).__name__}: {str(e)}"
            )


            with st.expander(
                "Technical Error"
            ):

                st.exception(e)


# ============================================================
# CURRENT DOCUMENT
# ============================================================

if st.session_state.indexed:

    st.success(
        f"📖 Current document: "
        f"**{st.session_state.document_name}**"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_input(
    "Ask a question about the document",
    placeholder=(
        "Example: Explain the confusion matrix "
        "of this paper. Is it good or bad?"
    ),
    key="document_question"
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True,
    key="ask_question_button"
)


if ask_button:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # =================================================
            # AUTOMATIC PREPARATION
            # =================================================

            if (
                st.session_state.rag is None
                or not st.session_state.indexed
            ):

                with st.spinner(
                    "📚 Preparing document for RAG..."
                ):

                    rag, stats = (
                        prepare_document(
                            pdf_path
                        )
                    )


                    st.session_state.rag = (
                        rag
                    )

                    st.session_state.document_path = (
                        str(pdf_path)
                    )

                    st.session_state.document_name = (
                        uploaded_file.name
                    )

                    st.session_state.indexed = (
                        True
                    )


                st.success(
                    "✅ Document prepared automatically."
                )


            # =================================================
            # ASK
            # =================================================

            with st.spinner(
                "🤖 Searching the document..."
            ):

                result = (
                    st.session_state.rag.ask(
                        question.strip()
                    )
                )


            st.session_state.last_result = (
                result
            )


        except Exception as e:

            st.error(
                f"Question answering failed: "
                f"{type(e).__name__}: {str(e)}"
            )

            with st.expander(
                "Technical Error"
            ):

                st.exception(e)


# ============================================================
# DISPLAY RAG RESULT
# ============================================================

result = (
    st.session_state.last_result
)


if result is not None:

    # ========================================================
    # ANSWER
    # ========================================================

    st.markdown(
        "## 🧠 Answer"
    )


    answer = result.get(
        "answer",
        ""
    )


    st.markdown(
        f"""
        <div class="answer-box">
        {answer}
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SOURCES
    # ========================================================

    sources = result.get(
        "sources",
        []
    )


    if sources:

        st.markdown(
            "## 📚 Sources"
        )


        for source in sources:

            if isinstance(
                source,
                dict
            ):

                modality = source.get(
                    "modality",
                    "unknown"
                )


                page = source.get(
                    "page",
                    "?"
                )


                st.markdown(
                    f"""
                    <div class="source-box">
                    📄 <b>{modality}</b>
                    &nbsp; • &nbsp;
                    Page <b>{page}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            else:

                st.markdown(
                    f"""
                    <div class="source-box">
                    📄 {source}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # RELATED IMAGES
    # ========================================================

    images = result.get(
        "images",
        []
    )


    if images:

        st.markdown(
            "## 🖼️ Related Document Pages"
        )


        columns = st.columns(
            min(
                len(images),
                3
            )
        )


        for index, image_path in enumerate(
            images
        ):

            column = columns[
                index % len(columns)
            ]


            with column:

                try:

                    st.image(
                        image_path,
                        caption=(
                            f"Document page "
                            f"{index + 1}"
                        ),
                        use_container_width=True
                    )

                except Exception:

                    st.warning(
                        f"Could not display: "
                        f"{image_path}"
                    )


# ============================================================
# ============================================================
# PLAGIARISM SECTION
# ============================================================
# ============================================================

st.divider()

st.markdown(
    "## 🔎 AI Plagiarism Checker"
)

st.caption(
    "Online source discovery + local AI similarity analysis."
)


# ============================================================
# CHECK PLAGIARISM BUTTON
# ============================================================

check_button = st.button(
    "🔍 CHECK PLAGIARISM",
    type="primary",
    use_container_width=True,
    key="check_plagiarism_button"
)


if check_button:

    try:

        # ====================================================
        # LOCAL PLAGIARISM MODEL
        # ====================================================

        with st.status(
            "Initializing plagiarism analysis...",
            expanded=True
        ) as status:

            st.write(
                "🧠 Loading local semantic model..."
            )


            checker = (
                PlagiarismChecker(
                    model_name="all-MiniLM-L6-v2",
                    similarity_threshold=0.55,
                    high_similarity_threshold=0.82
                )
            )


            st.write(
                "📄 Extracting document text..."
            )


            sentences = (
                checker.extract_pdf_sentences(
                    str(pdf_path)
                )
            )


            if not sentences:

                status.update(
                    label="No text found",
                    state="error"
                )


                st.error(
                    "Could not extract text from this PDF."
                )


                st.stop()


            passages = (
                checker.create_passages(
                    sentences
                )
            )


            st.write(
                f"Extracted "
                f"**{len(sentences)} sentences**."
            )


            st.write(
                f"Created "
                f"**{len(passages)} passages**."
            )


            status.update(
                label="Document text extracted",
                state="complete"
            )


        # ====================================================
        # WEB SEARCH
        # ====================================================

        st.markdown(
            "### 🌐 Searching Online Sources"
        )


        progress_bar = st.progress(
            0
        )


        progress_text = st.empty()


        web_search = (
            WebPlagiarismSearch(
                max_results_per_query=5,
                max_sources=15
            )
        )


        def update_progress(
            current,
            total
        ):

            if total <= 0:

                return


            progress = (
                current
                / total
            )


            progress_bar.progress(
                min(
                    progress,
                    1.0
                )
            )


            progress_text.write(
                f"Searching passage "
                f"{current} of {total}..."
            )


        source_documents = (
            web_search.search_document(
                passages,
                progress_callback=(
                    update_progress
                )
            )
        )


        progress_bar.progress(
            1.0
        )


        progress_text.write(
            f"Found "
            f"**{len(source_documents)} "
            f"candidate web sources**."
        )


        # ====================================================
        # PREPARE WEB SOURCES
        # ====================================================

        with st.spinner(
            "Preparing online source text..."
        ):

            source_passages = (
                web_search.prepare_for_ai(
                    source_documents,
                    checker
                )
            )


        st.write(
            f"Collected "
            f"**{len(source_passages)} "
            f"source passages**."
        )


        # ====================================================
        # PLAGIARISM COMPARISON
        # ====================================================

        with st.spinner(
            "🤖 Comparing document with online sources..."
        ):

            plagiarism_result = (
                checker.analyze_document(
                    str(pdf_path),
                    source_passages
                )
            )


        matches = (
            plagiarism_result.get(
                "matches",
                []
            )
        )


        overall_similarity = (
            plagiarism_result.get(
                "overall_similarity",
                0.0
            )
        )


        matched_percentage = (
            plagiarism_result.get(
                "matched_percentage",
                0.0
            )
        )


        high_matches = sum(
            1
            for match in matches
            if match.get("level")
            == "high"
        )


        moderate_matches = sum(
            1
            for match in matches
            if match.get("level")
            == "moderate"
        )


        # ====================================================
        # AI DETECTION
        # ====================================================

        with st.spinner(
            "🤖 Detecting AI-like writing locally..."
        ):

            ai_detector = (
                AIDetector(
                    threshold=0.65
                )
            )


            ai_result = (
                ai_detector.analyze_document(
                    str(pdf_path)
                )
            )


        ai_matches = (
            ai_result.get(
                "matches",
                []
            )
        )


        ai_percentage = (
            ai_result.get(
                "ai_percentage",
                0.0
            )
        )


        overall_ai_likelihood = (
            ai_result.get(
                "overall_ai_likelihood",
                0.0
            )
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        st.session_state.plagiarism_result = {
            "plagiarism": plagiarism_result,
            "ai": ai_result
        }


        # ====================================================
        # RESULT
        # ====================================================

        st.success(
            "✅ Analysis completed!"
        )


        st.markdown(
            "## 📊 Analysis Result"
        )


        # ====================================================
        # PLAGIARISM
        # ====================================================

        st.markdown(
            "### 🔎 Plagiarism / Text Similarity"
        )


        col1, col2, col3, col4 = (
            st.columns(4)
        )


        with col1:

            st.metric(
                "Potential Similarity",
                f"{overall_similarity:.2f}%"
            )


        with col2:

            st.metric(
                "Matched Content",
                f"{matched_percentage:.2f}%"
            )


        with col3:

            st.metric(
                "🔴 High",
                high_matches
            )


        with col4:

            st.metric(
                "🟡 Moderate",
                moderate_matches
            )


        # ====================================================
        # AI
        # ====================================================

        st.markdown(
            "### 🤖 AI Content Detection"
        )


        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            st.metric(
                "AI-Likelihood",
                f"{overall_ai_likelihood:.2f}%"
            )


        with col2:

            st.metric(
                "AI-like Content",
                f"{ai_percentage:.2f}%"
            )


        with col3:

            st.metric(
                "AI-like Passages",
                len(ai_matches)
            )


        st.caption(
            "🔴 Red = strong text similarity  •  "
            "🟡 Yellow = moderate text similarity  •  "
            "🔵 Blue = AI-like writing likelihood"
        )


        # ====================================================
        # HIGHLIGHTED PDF
        # ====================================================

        st.divider()


        st.markdown(
            "## 🖍️ Highlighted PDF"
        )


        output_pdf = (
            OUTPUT_DIR
            / f"{pdf_path.stem}"
            "_analysis_highlighted.pdf"
        )


        with st.spinner(
            "Highlighting detected passages..."
        ):

            highlighter = (
                PDFHighlighter()
            )


            highlight_result = (
                highlighter.create_highlighted_pdf(
                    input_pdf=str(
                        pdf_path
                    ),

                    output_pdf=str(
                        output_pdf
                    ),

                    matches=matches,

                    ai_matches=ai_matches
                )
            )


        # ====================================================
        # HIGHLIGHT COUNTS
        # ====================================================

        st.write(
            f"🔴 **Plagiarism highlights:** "
            f"{highlight_result.get('plagiarism_highlights', 0)}"
        )


        st.write(
            f"🔵 **AI highlights:** "
            f"{highlight_result.get('ai_highlights', 0)}"
        )


        # ====================================================
        # LEGEND
        # ====================================================

        st.markdown(
            """
            **Highlight Legend**

            🔴 **Red** → Strong plagiarism/text similarity

            🟡 **Yellow** → Moderate plagiarism/text similarity

            🔵 **Blue** → AI-like writing likelihood
            """
        )


        # ====================================================
        # PDF VIEWER
        # ====================================================

        if output_pdf.exists():

            st.pdf(
                str(output_pdf),
                height=850
            )


            with open(
                output_pdf,
                "rb"
            ) as f:

                pdf_bytes = f.read()


            st.download_button(
                label=(
                    "⬇️ Download Highlighted PDF"
                ),

                data=pdf_bytes,

                file_name=(
                    f"{pdf_path.stem}"
                    "_analysis_highlighted.pdf"
                ),

                mime="application/pdf",

                use_container_width=True,

                key="download_analysis_pdf"
            )


        # ====================================================
        # SIMILAR CONTENT
        # ====================================================

        st.divider()


        st.markdown(
            "## 📌 Detected Similar Content"
        )


        if not matches:

            st.success(
                "No potentially similar passages "
                "were detected from the web sources found."
            )


        else:

            sorted_matches = sorted(
                matches,
                key=lambda x: x.get(
                    "similarity",
                    0
                ),
                reverse=True
            )


            for i, match in enumerate(
                sorted_matches,
                start=1
            ):

                score = (
                    match.get(
                        "similarity",
                        0
                    )
                    * 100
                )


                level = match.get(
                    "level",
                    "moderate"
                )


                if level == "high":

                    title = (
                        f"🔴 HIGH — "
                        f"Match {i} — "
                        f"{score:.1f}%"
                    )

                else:

                    title = (
                        f"🟡 MODERATE — "
                        f"Match {i} — "
                        f"{score:.1f}%"
                    )


                with st.expander(
                    title
                ):

                    st.write(
                        f"**Your PDF page:** "
                        f"{match.get('page', 'N/A')}"
                    )


                    st.write(
                        f"**Similarity:** "
                        f"{score:.2f}%"
                    )


                    st.write(
                        f"**Semantic similarity:** "
                        f"{match.get('semantic_similarity', 0) * 100:.2f}%"
                    )


                    st.write(
                        f"**Text overlap:** "
                        f"{match.get('lexical_similarity', 0) * 100:.2f}%"
                    )


                    if match.get("url"):

                        st.markdown(
                            f"**Source:** "
                            f"[{match.get('source', 'Online Source')}]"
                            f"({match['url']})"
                        )


                    st.markdown(
                        "#### Your Content"
                    )


                    st.write(
                        match.get(
                            "text",
                            ""
                        )
                    )


                    st.markdown(
                        "#### Similar Online Content"
                    )


                    st.write(
                        match.get(
                            "reference_text",
                            ""
                        )
                    )


        # ====================================================
        # AI MATCHES
        # ====================================================

        st.divider()


        st.markdown(
            "## 🤖 AI-like Content"
        )


        if not ai_matches:

            st.success(
                "No passages crossed the AI-likelihood threshold."
            )

        else:

            sorted_ai = sorted(
                ai_matches,
                key=lambda x: x.get(
                    "ai_probability",
                    0
                ),
                reverse=True
            )


            for i, match in enumerate(
                sorted_ai,
                start=1
            ):

                score = (
                    match.get(
                        "ai_probability",
                        0
                    )
                    * 100
                )


                level = match.get(
                    "level",
                    "moderate"
                )


                if level == "high":

                    title = (
                        f"🔵 HIGH AI-LIKELIHOOD — "
                        f"Passage {i} — "
                        f"{score:.1f}%"
                    )

                else:

                    title = (
                        f"🔵 MODERATE AI-LIKELIHOOD — "
                        f"Passage {i} — "
                        f"{score:.1f}%"
                    )


                with st.expander(
                    title
                ):

                    st.write(
                        f"**PDF page:** "
                        f"{match.get('page', 'N/A')}"
                    )


                    st.write(
                        f"**AI-likelihood:** "
                        f"{score:.2f}%"
                    )


                    st.write(
                        match.get(
                            "text",
                            ""
                        )
                    )


    except Exception as e:

        st.error(
            f"Plagiarism analysis failed: "
            f"{type(e).__name__}: {str(e)}"
        )


        with st.expander(
            "Technical Error"
        ):

            st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Multimodal RAG + Local AI Plagiarism Analysis"
)