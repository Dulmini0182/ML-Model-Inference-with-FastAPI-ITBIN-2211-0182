Sentiment Analysis API

This project implements a **REST API** for sentiment analysis using **FastAPI**.  
It loads a pre-trained **TF-IDF + Logistic Regression** model (`model.pkl`) to classify text sentiment and optionally provide confidence scores.  

Features
- Fast and lightweight API built with **FastAPI**  
- **Single** and **batch** text predictions  
- **Health check** endpoint to verify service status  
- **Model info** endpoint for metadata  
- Input validation with **Pydantic**  

Installation

1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>


