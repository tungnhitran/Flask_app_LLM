# Watson AI Project

A basic Flask-based application that integrates with IBM Watson AI to provide language model capabilities. This project uses models like Llama 3.2 and Granite for text generation.

![Demo GIF](Screen Recording 2025-10-19 at 9.47.43 pm.gif)

## Features

- Integration with IBM Watson AI (watsonx)
- Support for multiple LLM models (Llama 3.2, Granite)
- Web interface built with Flask

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-directory>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root with your Watson AI credentials:

```
WATSONX_URL=https://au-syd.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_APIKEY=your_api_key_here
```

You can use `.env.example` as a template:
```bash
cp .env.example .env
```

Then edit `.env` with your actual credentials.

> **⚠️ Security Note:** Never commit the `.env` file to version control. It's included in `.gitignore` to protect your credentials.

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Available Models

- **Llama 3.2 (90B Vision Instruct):** `meta-llama/llama-3-2-90b-vision-instruct`
  - Advanced vision and text capabilities
  
- **Granite (3.2B Instruct):** `ibm/granite-3-2b-instruct`
  - Lightweight model for faster inference

## Usage

### Web Interface

Navigate to `http://localhost:5000` to access the web interface where you can interact with the models.

## Security

- Credentials are stored in `.env` and excluded from version control
- Never share your `.env` file or commit it to the repository
- Regenerate API keys if they are accidentally exposed
- Use `.env.example` as a template for other developers

## Dependencies

Main dependencies include:
- `flask` - Web framework
- `ibm-watsonx-ai` - IBM Watson AI SDK
- `python-dotenv` - Environment variable management

See `requirements.txt` for a complete list.

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Support

The LLM Models are deployed from IBM Watsonx AI

## Authors

(Alex) Tung Nhi TRAN