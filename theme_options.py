"""
Theme options for the Active Learning Dashboard.
You can easily switch between different color schemes here.
"""

# Available theme options
THEMES = {
    "light_gray": {
        "name": "Light Gray (Default)",
        "main_bg": "#f8f9fa",
        "card_bg": "#ffffff",
        "nav_info_bg": "#e8f4fd",
        "nav_info_border": "#b3d9ff",
        "primary_color": "#1f77b4",
        "text_color": "#262730"
    },
    
    "clean_white": {
        "name": "Clean White",
        "main_bg": "#ffffff",
        "card_bg": "#f8f9fa",
        "nav_info_bg": "#f0f8ff",
        "nav_info_border": "#d1ecf1",
        "primary_color": "#007bff",
        "text_color": "#212529"
    },
    
    "soft_blue": {
        "name": "Soft Blue",
        "main_bg": "#f0f8ff",
        "card_bg": "#ffffff",
        "nav_info_bg": "#e6f3ff",
        "nav_info_border": "#b3d9ff",
        "primary_color": "#0066cc",
        "text_color": "#1a1a1a"
    },
    
    "warm_beige": {
        "name": "Warm Beige",
        "main_bg": "#faf8f5",
        "card_bg": "#ffffff",
        "nav_info_bg": "#f5f2ed",
        "nav_info_border": "#e6ddd4",
        "primary_color": "#8b4513",
        "text_color": "#2c2c2c"
    },
    
    "mint_green": {
        "name": "Mint Green",
        "main_bg": "#f0fff4",
        "card_bg": "#ffffff",
        "nav_info_bg": "#e8f5e8",
        "nav_info_border": "#c3e6cb",
        "primary_color": "#28a745",
        "text_color": "#1a1a1a"
    },
    
    "dark_mode": {
        "name": "Dark Mode",
        "main_bg": "#2e3440",
        "card_bg": "#3b4252",
        "nav_info_bg": "#434c5e",
        "nav_info_border": "#5e81ac",
        "primary_color": "#88c0d0",
        "text_color": "#eceff4"
    }
}

def get_theme_css(theme_name="light_gray"):
    """Generate CSS for the specified theme."""
    theme = THEMES.get(theme_name, THEMES["light_gray"])
    
    return f"""
    <style>
    /* Main content area background */
    .main .block-container {{
        background-color: {theme['main_bg']};
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    .stApp {{
        background-color: {theme['main_bg']};
    }}
    
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {theme['primary_color']};
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    .metric-card, .config-section {{
        background-color: {theme['card_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid {theme['primary_color']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .experiment-card {{
        background-color: {theme['card_bg']};
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    
    .nav-info {{
        background-color: {theme['nav_info_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid {theme['nav_info_border']};
    }}
    
    .dataset-info {{
        background-color: {theme['nav_info_bg']};
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid {theme['nav_info_border']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Status colors */
    .status-running {{
        color: #28a745;
        font-weight: bold;
    }}
    
    .status-idle {{
        color: #6c757d;
        font-weight: bold;
    }}
    
    .status-error {{
        color: #dc3545;
        font-weight: bold;
    }}
    
    /* Info/success/warning boxes styling */
    .stAlert {{
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Metrics styling */
    [data-testid="metric-container"] {{
        background-color: {theme['card_bg']};
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }}
    
    /* Form elements styling */
    .stSelectbox > div > div {{
        background-color: {theme['card_bg']};
        border-radius: 6px;
    }}
    
    .stSlider > div > div {{
        background-color: {theme['card_bg']};
        border-radius: 6px;
        padding: 0.5rem;
    }}
    
    /* Text color for dark themes */
    {"color: " + theme['text_color'] + ";" if theme_name == "dark_mode" else ""}
    </style>
    """

def get_available_themes():
    """Get list of available theme names and display names."""
    return [(key, theme["name"]) for key, theme in THEMES.items()]