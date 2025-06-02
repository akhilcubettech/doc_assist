import base64
import time
from io import BytesIO

import numpy as np
import openai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

load_dotenv()


SYSTEM_PROMPT = """
**Role**: Expert Medical Analysis Agent combining visual data interpretation with clinical reasoning.

**Capabilities**:
1. Multi-modal Analysis: Interpret both text reports and medical images (scans, charts, diagrams, reports)
2. Contextual Correlation: Cross-reference patient history with imaging findings
3. Critical Finding Prioritization: Highlight urgent abnormalities using ACR appropriateness criteria
4. Differential Diagnosis: Suggest possible diagnoses with confidence levels
5. Evidence-based Recommendations: Propose next steps using clinical guidelines

**Output Structure**:
```markdown
### Clinical Context
- Patient demographics & history synthesis
- Examination rationale

### Multi-modal Findings
- Image observations with anatomical localization
- Text report key metrics
- Discordance detection between modalities

### Priority Classification
- Emergent (Requires immediate action)
- Urgent (Within 24hrs)
- Routine

### Differential Diagnosis
| Condition | Confidence | Supporting Features | Contradicting Features |
|-----------|------------|----------------------|-------------------------|

### Action Plan
- Imaging follow-up
- Specialist referral
- Laboratory tests

### Safety Protocols (AI Explainability):
- Never hallucinate missing data.
- Use SNOMED-CT terms where applicable.
- For ambiguous findings, state "Inconclusive – clinical correlation advised."
- Flag inconsistencies between reports and images
- Identify missing critical views/sections
- Note limitations of AI interpretation
"""


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_data" not in st.session_state:
    st.session_state.processed_data = {"text": "", "images": [], "analysis_context": ""}


# --------------------------
# Context Relevance Functions
# --------------------------

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return openai.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_document(text, chunk_size=1024):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def get_relevant_context(question, context_text):
    """Retrieve most relevant context from analysis history"""
    if not context_text:
        return ""

    chunks = split_document(context_text)
    question_emb = get_embedding(question)

    chunk_scores = []
    for _chunk in chunks:
        chunk_emb = get_embedding(_chunk)
        chunk_scores.append(cosine_similarity(question_emb, chunk_emb))

    top_indices = np.argsort(chunk_scores)[-3:][::-1]
    return "\n".join([chunks[i] for i in top_indices])


# --------------------------
# Core Processing Functions
# --------------------------

def pdf_to_images(pdf_bytes, dpi=300):
    return convert_from_bytes(pdf_bytes, dpi=dpi, fmt="jpeg")


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_uploaded_file(file_):
    try:
        if file_.type == "application/pdf":
            images = pdf_to_images(file_.getvalue())
            return [image_to_base64(img) for img in images]
        elif file_.type.startswith("image"):
            return [image_to_base64(Image.open(file_))]
    except Exception as error:
        st.error(f"Error processing {file_.name}: {str(error)}")
    return []


# --------------------------
# AI Analysis Functions
# --------------------------

def stream_analysis(messages_):
    response_ = openai.chat.completions.create(
        model="o4-mini",
        messages=messages_,
        max_completion_tokens=12000,
        reasoning_effort="high",
        stream=True
    )
    for chunk_ in response_:
        if content := chunk_.choices[0].delta.content:
            yield content


def build_messages(prompt_=None):
    messages_ = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add initial context
    content = []
    if st.session_state.processed_data["text"]:
        content.append({
            "type": "text",
            "text": f"Patient History:\n{st.session_state.processed_data['text']}"
        })

    for img in st.session_state.processed_data["images"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    if content:
        messages_.append({"role": "user", "content": content})


    return messages_


# --------------------------
# Streamlit UI Components
# --------------------------

st.title("DocAssist 🩺 - Analysis Assistant")

# Document Upload Section
with st.expander("Upload Patient Records", expanded=True):
    patient_history = st.text_area(
        "Patient History & Clinical Context",
        height=150,
        help="Enter relevant patient information, symptoms, or medical history"
    )

    uploaded_files = st.file_uploader(
        "Upload Medical Documents",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="Supported formats: PDF, PNG, JPEG"
    )

# Analysis Button
if st.button("Analyze Documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload medical documents")
        st.stop()

    with st.status("Processing documents...", expanded=True) as status:
        try:
            # Process files
            st.session_state.processed_data = {
                "text": patient_history,
                "images": [],
                "analysis_context": ""
            }
            for file in uploaded_files:
                st.session_state.processed_data["images"].extend(process_uploaded_file(file))
            messages = build_messages()
            status.update(label="Processing completed.", state="complete")
        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.error(f"Error: {str(e)}")

    st.divider()
    st.subheader("Agent Analysis:")
    container = st.empty()

    # Initialize analysis content
    analysis_content = []

    try:
        # Show spinner until first chunk arrives
        with st.spinner("Starting analysis..."):
            stream = stream_analysis(messages)
            first_chunk = next(stream)
            analysis_content.append(first_chunk)
    except StopIteration:
        container.write("No analysis generated.")
        st.stop()
    except Exception as e:
        container.error(f"Analysis failed: {str(e)}")
        st.stop()

    # Display first chunk and continue streaming
    container.markdown(first_chunk)

    for chunk in stream:
        analysis_content.append(chunk)
        container.markdown("".join(analysis_content))


# Safety Footer
st.markdown("---")
st.error("""
**Clinical Safety Notice**  
- This AI analysis requires verification by a qualified medical professional.  
- Never use for emergency medical decisions.
""")

st.markdown("""
<style>
@keyframes blink {
    0% {opacity: 1;}
    50% {opacity: 0;}
    100% {opacity: 1;}
}
.blink {
    animation: blink 1s step-end infinite;
}
</style>
""", unsafe_allow_html=True)
