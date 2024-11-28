---
title: XAI Testing
emoji: 📉
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 5.7.0
app_file: app.py
pinned: false
---

# xAI-testing

## Setup

1. Clone the repository:
```
git clone https://github.com/142CodeGreen/xAI-testing.git
cd xAI-testing
```

2. Install the required packages:
```
pip install --upgrade -r requirements.txt
```

3. Export API keys. NVIDIA_API_KEY is for NVIDIA NIM, while OpenAI API Key is needed for Nemo Guardrails. 
```
export XAI_API_KEY="your-api-key-here"
echo $XAI_API_KEY

export OPENAI_API_KEY="your-api-key-here"
echo $OPENAI_API_KEY

```

4. Run the app.py:
```
python3 app.py
```
