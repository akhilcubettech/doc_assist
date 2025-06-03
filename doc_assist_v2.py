import base64
import time
from io import BytesIO
import bcrypt
import numpy as np
import openai
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from pdf2image import convert_from_bytes

load_dotenv()


st.set_page_config(
    page_title="DocAssist - Medical Analysis",
    page_icon="🩺",
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed
)

# Initialize authentication state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.auth_failed = False
    st.session_state.username = ""

# Create a separate login page if not authenticated
if not st.session_state.authenticated:
    st.title("Medical Analysis Portal 🔒")

    # Use form to prevent immediate validation
    with st.form("login_form"):
        username = st.text_input("Username", key="username_input")
        password = st.text_input("Password", type="password", key="password_input")
        submit_button = st.form_submit_button("Login", type="primary")

        if submit_button:
            try:
                # Check if secrets are loaded
                if "passwords" not in st.secrets:
                    st.error("Authentication system not configured")
                    st.session_state.auth_failed = True
                else:
                    # Check credentials
                    if username in st.secrets["passwords"]:
                        hashed_password = st.secrets.passwords[username]
                        if bcrypt.checkpw(password.encode(), hashed_password.encode()):
                            st.session_state.authenticated = True
                            st.session_state.auth_failed = False
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.session_state.auth_failed = True
                    else:
                        st.session_state.auth_failed = True
            except Exception as e:
                st.error(f"Authentication error: {str(e)}")
                st.session_state.auth_failed = True

        if st.session_state.auth_failed:
            st.error("Invalid username or password")

    # Stop execution if not authenticated
    st.stop()

# If authenticated, show the main application
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.auth_failed = False
        st.session_state.username = ""
        st.rerun()


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
```markdown
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
