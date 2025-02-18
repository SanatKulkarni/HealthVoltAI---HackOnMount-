# HealthVoltAI - Revolutionizing Healthcare Documentation & Analysis (HackOnMount Project)
![Logo](https://github.com/user-attachments/assets/5e060701-b6dc-40d2-93fa-aaa533d6e720)

## Watch the Video

[![YouTube Video](https://img.youtube.com/vi/sggMvVYPiQQ/0.jpg)](https://www.youtube.com/watch?v=sggMvVYPiQQ)

Click the thumbnail above to watch the video.

**HealthVoltAI** is an innovative, AI-powered healthcare document processing system designed to streamline documentation and analysis for healthcare providers and insurers, making critical data more accessible and actionable. This project was developed by **Team int21h** for the HackOnMount hackathon.

## Problem Statement

Healthcare documentation is often a complex, time-consuming, and error-prone process.  Providers and insurers struggle with:

*   **Manual Data Entry:**  Extracting information from a variety of sources (handwritten notes, digital documents, faxes, etc.) is tedious and inefficient.
*   **Data Silos:**  Information is scattered across different systems, making it difficult to get a complete picture of a patient's history or claim.
*   **Complex Analysis:**  Analyzing large volumes of medical data to identify trends, patterns, and risks is challenging and requires specialized expertise.

**Our Solution: HealthVolt.AI - All-in-One Medical Documents to Decisions. Instantly.**

## Key Features

HealthVolAI addresses these challenges with three core capabilities:

1.  **Intelligent Document Parsing and Extraction:**
    *   AI-powered engine automatically extracts structured data (tables, forms, invoices) and medical diagnoses from diverse document formats (handwritten, digital) with high accuracy using advanced OCR and NLP techniques.
    *   Converts unstructured medical terminology into standardized, actionable data.

2.  **Context-Aware Chatbot:**
    *   Leverages extracted data and powerful analytics to provide instant answers about patient records, policy details, and billing inquiries in natural language.
    *   Continuously learns from new medical terminology and document formats, improving accuracy and understanding over time.

3.  **Real-Time Analytics and Reporting:**
    *   Aggregates processed medical data to identify claim patterns, common diagnoses, treatment costs, and risks.
    *   Enables better policy design, risk assessment, and informed decision-making.

## How It Works

Here's a simplified overview of the HealthVoltAI processing pipeline:

1.  **User Input:** User uploads a medical document image or folder path.
2.  **Frontend Layer:** Processes input and sends it to the backend through an API.
3.  **Backend Processing:**
    *   **Image Pre-processing:** Employs techniques like grayscale conversion, contrast enhancement, and binarization.
    *   **Region of Interest (ROI) Location:** Utilizes PaddleOCR to identify areas containing provisional diagnosis or key medical information.
    *   **Cropping:** Precisely crops the identified ROI.
    *   **Handwritten Text Extraction:** Uses Qwen-VL-2B-Instruct model.
    *   **Text Processing:** Correction of spelling errors and abbreviations (using medical knowledge graph).
4.  **Output:** Structured data in Excel file is generated and made available for download.


## Technology Stack

HealthVoltAI is built on a modern and robust technology stack:

*   **Frontend:** Next.js with Tailwind CSS
*   **Backend:** Flask (Python) - API and Data Processing
*   **OCR:** PaddleOCR and DocTR
*   **Deep Learning:** PyTorch
*   **Image Processing:** OpenCV
*   **Chatbot and Language Model:** Qwen (integrated through LangChain)
*   **Data Manipulation and Analysis:** Pandas and NumPy
*   **Language Model Orchestration**: LangChain

## Use Cases

HealthVoltAI offers a wide range of applications, including:

*   **Accelerated Claims Processing:** Faster extraction of diagnostic codes and policy matching.
*   **Fraud Detection:** Identification of unusual billing patterns and duplicate claims.
*   **Improved Customer Service:** Policyholders can upload documents and receive immediate explanations of benefits through the chatbot, reducing call center volume.
*   **Risk Management:** Processing historical claim documents to create more accurate risk profiles.
*   **Regulatory Compliance:** Automated tracking and maintenance of required documentation.
*   **Targeted Wellness Programs:** Identifying high-cost claim categories to inform targeted wellness programs or coverage adjustments.
*   **Faster Dispute Resolution:** Enhanced data and analytics lead to quicker and more efficient resolution of disputed claims.

## Dependencies & Show Stoppers

*   **Complete historical tracking** preserves every version of medical documents with full context, making it easy to understand how diagnoses evolved over time
*   The AI **continuously learns** from new medical terminology and document formats it encounters, becoming more accurate and capable over time without manual updates
*   A **unified system** eliminates data silos and conflicting information by providing a single, authoritative source for all medical records and billing data
*   The **platform maintains strict HIPAA compliance** while providing lightning-fast access to data, balancing security with performance

## Team Members

*   **Sanat Kulkarni (Team Leader):** B.Tech (Computer Science & Engineering - Big Data & Analytics) - Year III
*   **Rachit Methwani:** B.Tech (Computer Science & Engineering - Big Data & Analytics) - Year III
*   **Shivansh Singhania:** B.Tech (Computer Science & Engineering - Big Data & Analytics) - Year III
*   **Aditya Chandra:** B.Tech (Computer Science & Engineering - AI & ML) - Year III

## Contact

For questions or inquiries, please contact [your email address].

## License

[Choose a license, e.g., MIT License.  You'll also need to create a LICENSE file in your repository.]
