# AI Fashion Color Recommender

## Overview

This is an AI-powered fashion color recommendation web application styled as a high-end e-commerce platform (inspired by Myntra/H&M). The application analyzes users' skin tones and undertones from uploaded face images using computer vision, then provides personalized clothing color recommendations. The UI features modern e-commerce aesthetics with animations, product cards, and a polished shopping experience.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit with embedded custom HTML/CSS
- **Styling Approach**: Custom CSS injected via `st.markdown()` for Myntra/H&M aesthetic
- **Design System**: 
  - Inter font family
  - Primary accent color: #ff3f6c (pink/red)
  - Card-based layouts with shadows and rounded corners
  - CSS animations (fade-in, slide-up, hover effects)
- **Navigation**: Session state-based page routing (Home, Style Analyzer, Color Guide, About)

### Backend Architecture
- **Runtime**: Python with Streamlit as the web framework
- **Image Processing Pipeline**:
  1. User uploads face image
  2. OpenCV Haar Cascade detects face region
  3. Extract facial skin region from detected area
  4. Convert to HSV color space
  5. Use KMeans clustering (scikit-learn) to find dominant skin color
  6. Classify skin tone (Light/Medium/Dark) and undertone (Warm/Cool/Neutral)
  7. Apply rule-based logic to recommend outfit colors

### State Management
- Streamlit session state manages:
  - Current page navigation
  - Analysis completion status
  - Uploaded image data

### AI/ML Components
- **Face Detection**: OpenCV Haar Cascade classifiers
- **Color Analysis**: KMeans clustering for dominant color extraction
- **Classification**: Rule-based logic for skin tone and undertone determination
- **Confidence Scoring**: Randomized percentage (82%-95%) for UX purposes

## External Dependencies

### Python Packages
| Package | Purpose |
|---------|---------|
| streamlit | Web application framework |
| opencv-python-headless | Face detection and image processing (headless for server deployment) |
| numpy | Numerical operations for image data |
| pillow | Image file handling and manipulation |
| scikit-learn | KMeans clustering for color analysis |

### External Resources
- Google Fonts API (Inter font family)
- Fashion image URLs for product card displays

### Development Configuration
- Server runs on port 5000
- Configured for 0.0.0.0 address binding (Replit compatibility)
- npm scripts available for development (`npm run dev`)