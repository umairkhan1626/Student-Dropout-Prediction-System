"""
EduGuard - Student Retention Portal
====================================
Module: styles.py
Description:
    Provides custom CSS styles, HTML badge generators, and UI layout components
    to render a modern, website-style appearance for the EduGuard portal:
    - Fixed Navy & Slate color palette
    - Modern elevated card styling with soft shadows
    - Color-coded risk badges (High, Medium, Low)
    - Clean typography and hidden Streamlit chrome
"""

import streamlit as st
from datetime import datetime


def inject_custom_css():
    """
    Injects custom CSS to style the Streamlit interface as a polished,
    modern SaaS-style web application.
    """
    custom_css = """
    <style>
        /* Import clean modern sans-serif font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* Root styling and font */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1e293b;
        }

        /* Hide default Streamlit header bar, deploy button, and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {display: none;}
        .stDeployButton {display: none;}

        /* Main app container background */
        .stApp {
            background-color: #f8fafc;
        }

        /* Top Brand Header Banner */
        .eduguard-header {
            background: linear-gradient(135deg, #0f2b48 0%, #1a4269 100%);
            color: #ffffff;
            padding: 1.25rem 2rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(15, 43, 72, 0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .eduguard-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .eduguard-logo-icon {
            font-size: 2rem;
            background: rgba(255, 255, 255, 0.15);
            padding: 6px 12px;
            border-radius: 8px;
        }
        .eduguard-title {
            font-size: 1.5rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0;
            color: #ffffff;
        }
        .eduguard-subtitle {
            font-size: 0.85rem;
            color: #94a3b8;
            margin: 2px 0 0 0;
        }
        .eduguard-date-badge {
            background: rgba(255, 255, 255, 0.12);
            color: #e2e8f0;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }

        /* KPI Metric Cards */
        .metric-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        }
        .metric-title {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f2b48;
            line-height: 1.2;
        }
        .metric-caption {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.35rem;
        }
        .metric-accent-bar {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
        }

        /* Content Card Container */
        .content-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            margin-bottom: 1.5rem;
        }

        /* Risk Badges */
        .badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-align: center;
        }
        .badge-high {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
        }
        .badge-medium {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
        }
        .badge-low {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
        }

        /* Status Badges */
        .badge-flagged {
            background-color: #eff6ff;
            color: #1d4ed8;
            border: 1px solid #bfdbfe;
        }
        .badge-contacted {
            background-color: #fdf4ff;
            color: #86198f;
            border: 1px solid #f5d0fe;
        }
        .badge-resolved {
            background-color: #f0fdf4;
            color: #15803d;
            border: 1px solid #bbf7d0;
        }

        /* Specialty Badges */
        .badge-academic {
            background-color: #e0f2fe;
            color: #0369a1;
            border: 1px solid #bae6fd;
        }
        .badge-financial {
            background-color: #ecfdf5;
            color: #047857;
            border: 1px solid #a7f3d0;
        }
        .badge-engagement {
            background-color: #fdf2f8;
            color: #be185d;
            border: 1px solid #fbcfe8;
        }

        /* Counselor Card */
        .counselor-card {
            background: #ffffff;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
            transition: all 0.2s ease;
        }
        .counselor-card:hover {
            box-shadow: 0 6px 14px rgba(0,0,0,0.08);
            border-color: #cbd5e1;
        }

        /* Styled Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        /* Button Styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_top_banner():
    """
    Renders the professional EduGuard top banner with brand title,
    subtitle, and current calendar date.
    """
    today_str = datetime.now().strftime("%B %d, %Y")
    banner_html = f"""
    <div class="eduguard-header">
        <div class="eduguard-brand">
            <div class="eduguard-logo-icon">🛡️</div>
            <div>
                <h1 class="eduguard-title">EduGuard</h1>
                <p class="eduguard-subtitle">Student Retention & Early Intervention Portal</p>
            </div>
        </div>
        <div class="eduguard-date-badge">
            📅 {today_str}
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def get_risk_badge(risk_score: float) -> str:
    """
    Returns an HTML badge string corresponding to the student's risk category:
    - High: Risk >= 0.60 (Red)
    - Medium: 0.30 <= Risk < 0.60 (Amber)
    - Low: Risk < 0.30 (Green)
    """
    if risk_score >= 0.60:
        return '<span class="badge badge-high">High Risk</span>'
    elif risk_score >= 0.30:
        return '<span class="badge badge-medium">Medium Risk</span>'
    else:
        return '<span class="badge badge-low">Low Risk</span>'


def get_status_badge(status: str) -> str:
    """
    Returns an HTML badge string for student case status:
    - Flagged (Blue)
    - Contacted (Purple)
    - Resolved (Green)
    """
    status_lower = str(status).lower()
    if "resolved" in status_lower:
        return f'<span class="badge badge-resolved">● Resolved</span>'
    elif "contacted" in status_lower:
        return f'<span class="badge badge-contacted">● Contacted</span>'
    else:
        return f'<span class="badge badge-flagged">● Flagged</span>'


def render_kpi_card(title: str, value: str, caption: str, accent_color: str = "#0f2b48"):
    """
    Renders a styled metric card with custom title, value, caption, and accent color.
    """
    card_html = f"""
    <div class="metric-card">
        <div class="metric-accent-bar" style="background-color: {accent_color};"></div>
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-caption">{caption}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
