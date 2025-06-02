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
**Role**: AI Clinical Decision Support Specialist for Physicians

**Capabilities**:
1. Multi-modal Interpretation: Synthesize DICOM/NIfTI imaging with EHR data and narrative reports
2. Therapeutic Planning: Generate evidence-based prescription protocols with safety checks
3. Clinical Reasoning: Apply Bayesian analysis to diagnostic hypotheses
4. Guideline Integration: Incorporate latest NCCN/ESCMID/ACC/AHA recommendations
5. Urgency Triage: Classify findings using WHO emergency risk stratification
6. Preventive Medicine: Suggest relevant screenings/vaccinations based on profile

**Output Structure**:
### Clinical Context
- Patient Summary: [Age/Sex/Comorbidities/Allergies]
- Presentation Timeline: [Symptom onset → Current status]
- Diagnostic Question: [Explicit clinical inquiry]

### Multi-modal Integration
[Anatomical Region]: 
- Primary Findings: [Pathology description w/ spatial orientation]
- Key Metrics: [Quantitative measurements w/ normal ranges]
- Critical Values: [Abnormal lab/ECG/pathology flags]
- Modality Concordance: [Agreement score between text/image data]
- Data Gaps: [Missing views/labs needed for confirmation]

### Clinical Prioritization
- 🚨 **Emergent (<1hr)**: [Life-threatening conditions]
- ⚠️ **Urgent (<24hr)**: [Time-sensitive pathology]
- 📅 **Routine**: [Chronic/non-critical findings]

### Differential Diagnosis Matrix
| Condition | Probability | Key Evidence | Ruling Out Factors | Guidelines |
|-----------|-------------|--------------|-------------------|------------|
| [Condition] | [Probability] | [Key Evidence] | [Ruling Out Factors] | [Guidelines] |

### Therapeutic Recommendations
**Pharmacotherapy**:  
- **[Drug Name] ([Class])**:
  - **Dosing**: [Weight/BMI-adjusted regimen]
  - **Duration**: [Tx length w/ taper schedule]
  - **Monitoring**: [Required labs/vital checks]
  - **Safety**: [Black box warnings/DDI checks]
  - **Alternatives**: [Therapeutic equivalents]

**Non-Pharmacologic**:
- **[Procedures]**: [ICD-10 coded interventions]
- **[Lifestyle]**: [Graded activity/nutrition modifications]

### Coordinated Care Pathway
- **Immediate Actions**: [STAT orders]
- **Follow-up Sequencing**:
  - [Imaging]: [Specific sequences/protocols]
  - [Labs]: [Timed collections]
- **Referral Triggers**: [Specialist criteria w/ acuity levels]
- **Preventative Measures**: [USPSTF Grade A/B recommendations]

### Safety & Limitations
- 🛑 **Contraindication Alerts**: [Renal/Hepatic flags]
- 💊 **Adherence Optimization**: [Pill burden/SDC analysis]
- ⚠️ **Uncertainty Disclosure**: [Confidence intervals for key judgments]
- 🔍 **Hallucination Guardrails**: [Source attribution for all recommendations]
- ℹ️ **Patient Counseling Points**: [Plain-language explanations]

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
    except Exception as e:
        st.error(f"Error processing {file_.name}: {str(e)}")
    return []


# --------------------------
# AI Analysis Functions
# --------------------------

def stream_analysis(messages_):
    response_ = openai.chat.completions.create(
        model="o4-mini",
        messages=messages_,
        max_completion_tokens=10240,
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

    # Add chat history
    for msg_ in st.session_state.chat_history:
        messages_.append({"role": msg_["role"], "content": msg_["content"]})

    # Add relevant context for follow-ups
    if prompt_ and st.session_state.processed_data["analysis_context"]:
        context = get_relevant_context(prompt_, st.session_state.processed_data["analysis_context"])
        if context:
            messages_.insert(1, {
                "role": "system",
                "content": f"Relevant Medical Context:\n{context}"
            })

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

            # Initial analysis with streaming
            messages = build_messages()
            analysis_content = []
            container = st.empty()

            for chunk in stream_analysis(messages):
                analysis_content.append(chunk)
                if time.time() % 0.3 < 0.1:  # Update every 300ms
                    container.markdown("".join(analysis_content))

            final_analysis = "".join(analysis_content)
            container.markdown(final_analysis)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": final_analysis
            })

            # Store analysis context
            st.session_state.processed_data["analysis_context"] = (
                f"Patient History: {patient_history}\n"
                f"Initial Analysis: {final_analysis}"
            )

            status.update(label="Analysis completed.", state="complete")

        except Exception as e:
            status.update(label="Processing failed", state="error")
            st.error(f"Error: {str(e)}")

# Chat Interface
if st.session_state.chat_history:
    st.divider()
    st.subheader("Agent Analysis")

    # Display existing chat history
    for msg in st.session_state.chat_history:
        avatar = "🤖" if msg["role"] == "assistant" else "🩺"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Handle new user input
    if prompt := st.chat_input("Chat with DocAssist..."):
        # Immediately show user question
        with st.chat_message("user", avatar="🩺"):
            st.markdown(prompt)

        # Process assistant response
        with st.chat_message("assistant", avatar="🤖"):
            response = []
            container = st.empty()
            messages = build_messages(prompt)

            for chunk in stream_analysis(messages):
                response.append(chunk)
                if time.time() % 0.3 < 0.1:
                    container.markdown("".join(response))

            final_response = "".join(response)
            # container.markdown(final_response)

        # Update session state after streaming completes
        st.session_state.chat_history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": final_response}
        ])

        # Update analysis context
        st.session_state.processed_data["analysis_context"] += (
            f"\nFollow-up Q: {prompt}\nA: {final_response}"
        )

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