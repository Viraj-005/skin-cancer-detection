import streamlit as st
from streamlit_option_menu import option_menu
import home
import detection
import visualizing
import train_model

# Set page configuration with wide layout, page title, and icon
st.set_page_config(layout="wide", page_title="SmartSkin Scan", page_icon="🩺")

# Custom CSS to disable scrolling in the sidebar and reduce empty space
st.markdown("""
            <style>
            
             .css-1iyw2u1 {
                    display: none;
                }
            
            </style>
            
            """, 
            unsafe_allow_html=True
            )

# MultiApp class to manage multiple applications
class MultiApp:
    def __init__(self):
        self.apps = []

    # Use for add a new application to the list
    def add_app(self, title, func):
        self.apps.append({"title": title, "function": func})

    # Run the selected application
    def run(self):
        with st.sidebar:
            selected_app = option_menu(
                menu_title="🩺 SmartSkin Scan",
                options=[app["title"] for app in self.apps],
                icons=["house", "camera", "bar-chart-line", "tools"],
                menu_icon="ss",
                default_index=0, # Default selected app
                styles={
                    "container": {
                        "padding": "10px 5px", 
                        "background-color": "rgba(255, 255, 255, 0.1)",
                        "backdrop-filter": "blur(10px)",
                        "border-radius": "10px",
                        "box-shadow": "0 4px 8px 0 rgba(0, 0, 0.2, 0.2)",
                    },
                    "icon": {
                        "color": "#ffffff",  
                        "font-size": "20px",  
                        "margin-right": "10px",
                    }
                }
            )
            # Give padding for the main content container
            st.markdown('<style>div.block-container{padding-top:3.2rem;}</style>', unsafe_allow_html=True)
            
            # Add container for display the social media links
            with st.container():
                st.markdown(
                    """
                    <hr style="border:0; border-top:1px solid #ccc;" />
                    <div style="background-color: rgba(255, 255, 255, 0.1); padding: 10px; border-radius: 10px; backdrop-filter: blur(10px); box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);">
                    <p><strong>Follow me on:</strong></p>
                    <ul style="list-style-type: none; padding-left: 0;">
                        <li>
                            <img src="https://cdn.icon-icons.com/icons2/936/PNG/512/github-logo_icon-icons.com_73546.png" alt="GitHub" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 5px;">
                            <a href="https://github.com/Viraj-005" target="_blank">@virajInduruwa</a>
                        </li>
                        <li style="margin-top: 20px;">
                            <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/linkedin_logo_icon_170234.png" alt="LinkedIn" style="width: 20px; height: 20px; vertical-align: middle; margin-right: 5px;">
                            <a href="https://www.linkedin.com/in/viraj-induruwa" target="_blank">Viraj Induruwa</a>
                        </li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display the version of the web-app
                st.markdown(
                    """
                    <div style="text-align: center; padding: 10px 0; margin-bottom: -100px;">
                    <p style="color: #0F0F0F; background-color: rgba(255, 255, 255, 0.9); padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 20px; backdrop-filter: blur(10px); box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);">
                        SmartSkin Scan - Version 1.5</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )

        # Run the selected app from the sidebar menu
        for app in self.apps:
            if app["title"] == selected_app:
                app["function"]()

# Add the individual apps to the MultiApp instance
app = MultiApp()
app.add_app("Home", home.app)
app.add_app("Detection", detection.app)
app.add_app("Visualizing", visualizing.app)
app.add_app("Train Model", train_model.app)
app.run()
