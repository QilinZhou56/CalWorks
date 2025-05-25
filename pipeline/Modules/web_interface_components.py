import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document as WordDoc


# Interface 1: County_level Map with classified CalWORKs metrics
# Load county-level outcome data in the Asset folder
# Change the Path to where you stored the county GIS information csv
MAP_DF = pd.read_csv("/content/drive/MyDrive/LLM/CalWorks/Vector Database/Asset/[GIS]ca_counties_outcome.csv")

# The 15 outcome metrics you want to visualize
OUTCOME_METRICS = [
    'Ancillary_Access_Rate',
    'Education_and_Skills_Dev_Rate',
    'Employment_Rate',
    'Engagement_Rate',
    'Exits_With_Earnings',
    'Family_Stabilization_to_WTW_Eng',
    'First_Activity_Rate',
    'OCAT_Appraisal_to_Next_Activity',
    'OCAT_Timeliness_Rate',
    'PostCWEmployment',
    'Reentry_After_Exit_with_Earning',
    'Reentry',
    'Sanction_Rate',
    'Sanction_Resolution_Rate',
    'Orientation_Attendance_2024'
]

COUNTIES = sorted(MAP_DF['County'].unique())

# Plot callback with optional county filter
def plot_outcome_map(metric, selected_counties):
    df = MAP_DF.copy()
    if selected_counties:
        df = df[df['County'].isin(selected_counties)]

    color_map = {
    "low": "#d62728",     # red → poor outcome
    "medium": "#ff7f0e",  # orange → moderate
    "high": "#2ca02c"     # green → favorable
}
    fig = go.Figure()

    for label in df[metric].dropna().unique():
        mask = df[metric] == label
        fig.add_trace(go.Scattermapbox(
            lat=df.loc[mask, 'INTPTLAT'],
            lon=df.loc[mask, 'INTPTLON'],
            text=df.loc[mask, 'County'],
            mode='markers',
            name=label.title(),
            marker=go.scattermapbox.Marker(
                size=8,
                color=color_map.get(label.lower(), '#636EFA')
            ),
            hovertemplate=(
                "<b>County</b>: %{text}<br>"
                f"<b>Category</b>: {label.title()}"
            )
        ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center={"lat": 37.5, "lon": -119.5},
            zoom=5.2,  # Or try zoom=4.5
            style="open-street-map"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{metric.replace('_',' ').title()} Classification by County"
    )

    return fig

# Interface 2: Multi-modal interface that allows users to upload documents or images
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return ""

    name = uploaded_file.name.lower()

    # PDF: use PyMuPDF
    if name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n\n".join([page.get_text() for page in doc])

    # Word: .docx
    elif name.endswith(".docx"):
        doc = WordDoc(uploaded_file)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    # Excel / CSV
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        return df.to_csv(index=False)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df.to_csv(index=False)

    # Image: OCR
    elif name.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)

    return "❌ Unsupported file type."