# LinkedIn Post Draft

I just shipped a full-stack AI text detection project from model training to live deployment.

Live app: https://frontend-navy-seven-60.vercel.app

This started as a fine-tuned RoBERTa classifier for detecting AI-generated writing, but I wanted to take it beyond a notebook and turn it into something people can actually use.

What I built:

- fine-tuned a RoBERTa sequence classification model
- packaged the model in Hugging Face Transformers format
- built a FastAPI inference backend with typed request/response schemas
- added long-document analysis using overlapping 512-token model windows
- grouped neighboring chunks with unusually high AI probability so users can inspect suspicious regions
- added CPU dynamic quantization to make free-tier inference lighter
- containerized the backend with Docker
- deployed the backend on Hugging Face Spaces
- built a smooth Vite React + TypeScript frontend
- deployed the frontend on Vercel
- documented the architecture, model behavior, deployment process, and tradeoffs

One of the biggest lessons was that the model is only one part of the system. A usable ML product also needs API design, deployment constraints, frontend state handling, secret hygiene, model loading strategy, performance tradeoffs, and responsible-use framing.

For example, RoBERTa has a fixed token limit, so long documents cannot just be pushed through the model in one shot. Instead of truncating the text, I split it into overlapping token windows, scored every chunk, aggregated the document-level result, and surfaced high-risk grouped regions.

I also chose FastAPI over heavier backend frameworks because this service is focused on ML inference, not a database-heavy web app. Hugging Face Spaces made sense for the model-serving container, while Vercel was a better fit for the static React frontend.

This project helped me connect model training, backend engineering, frontend product design, and cloud deployment into one end-to-end system.

Tech stack:
RoBERTa, Hugging Face Transformers, PyTorch, FastAPI, Docker, React, TypeScript, Vite, Hugging Face Spaces, Vercel

GitHub: https://github.com/AtharvaMaik/AIDetect
Backend: https://clashking-aidetect.hf.space

I am especially interested in roles where I can work across machine learning, backend systems, and product engineering to turn models into real user-facing tools.

#MachineLearning #AI #NLP #FastAPI #React #HuggingFace #PyTorch #FullStackDevelopment #MLOps #SoftwareEngineering
