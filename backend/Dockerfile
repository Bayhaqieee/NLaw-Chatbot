FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Pre-download NLTK data (needed by METEOR metric) at build time
RUN python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"

# Pre-cache HuggingFace evaluate builder scripts at build time
# This prevents the "Downloading builder script" spam on every evaluation call
RUN python -c "\
import evaluate; \
evaluate.load('sacrebleu'); \
evaluate.load('rouge'); \
evaluate.load('meteor'); \
evaluate.load('bertscore'); \
print('All evaluate metrics pre-cached.')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
