## 🚀 Overview

**Visual Product Search** is a powerful **Content-Based Image Retrieval (CBIR)** system built for the fashion domain.  
It enables users to find visually similar fashion products using **image queries**, powered by a state-of-the-art **CLIP (ViT-L/14)** embedding model.

The system understands the semantic and visual characteristics of fashion items and retrieves the closest matches from a large product database—making it ideal for recommendation engines, e-commerce search, and style discovery.

### Key Features


-   **Advanced AI Model:** Fine-tuned version of `openai/clip-vit-large-patch14` for superior feature extraction.
    
-   **Specialized Dataset:** Trained on the `paramaggarwal/fashion-product-images-dataset` to ensure high accuracy in fashion categorization.
    
-   **Web Interface:** Clean, responsive frontend built with HTML and CSS.
    
-   **Robust Backend:** Powered by Flask to handle API requests and model inference.
    
-   **Containerized:** Includes Docker support for easy deployment.
    

## 🛠️ Technology Stack


-   **Frontend:** HTML5, CSS3
    
-   **Backend:** Python, Flask
    
-   **Machine Learning:** PyTorch, Hugging Face Transformers
    
-   **Model Architecture:** CLIP (Contrastive Language-Image Pre-Training) ViT-Large-Patch14
    
-   **Deployment:** Docker
    

## 💻 Installation & Setup


Follow these steps to run the project locally on your machine.

### 1\. Clone the Repository

# 

    git clone https://github.com/mdmohsin212/Visual-Product-Search.git
    

### 2\. Create a Virtual Environment (Recommended)

# 

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    

### 3\. Install Dependencies

# 

    pip install -r requirements.txt
    

### 4\. Run the Application

# 

    python app.py
    

The application will start on `http://127.0.0.1:5000/` (or the port specified in your console).

## 🐳 Docker Support

# 

You can also run this application using Docker to avoid environment configuration issues.

1.  **Build the Image:**
    
        docker build -t visual-product-search .
        
    
2.  **Run the Container:**
    
        docker run -p 5000:5000 visual-product-search
        
    

## 📊 Model Performance


The model has been rigorously validated using standard Information Retrieval metrics.

### Validation Recall Metrics


Recall measures the proportion of relevant items retrieved in the top _k_ results.

| Metric | Score | Metric | Score |
| --- | --- | --- | --- |
| **Recall@1** | 0.2602 | **Recall@40** | 0.9232 |
| **Recall@5** | 0.5644 | **Recall@50** | 0.9410 |
| **Recall@10** | 0.7008 | **Recall@60** | 0.9534 |
| **Recall@15** | 0.7770 | **Recall@70** | 0.9644 |
| **Recall@20** | 0.8270 | **Recall@80** | 0.9700 |
| **Recall@25** | 0.8626 | **Recall@90** | 0.9744 |
| **Recall@30** | 0.8896 | **Recall@100** | 0.9780 |
| **Recall@35** | 0.9092 | **Recall@200** | 0.9922 |

### Validation NDCG Metrics

Normalized Discounted Cumulative Gain (NDCG) accounts for the position of relevant items in the ranking.

| Metric | Score | Metric | Score |
| --- | --- | --- | --- |
| **NDCG@1** | 0.2602 | **NDCG@40** | 0.5150 |
| **NDCG@5** | 0.4191 | **NDCG@50** | 0.5183 |
| **NDCG@10** | 0.4632 | **NDCG@60** | 0.5204 |
| **NDCG@15** | 0.4834 | **NDCG@70** | 0.5222 |
| **NDCG@20** | 0.4952 | **NDCG@80** | 0.5231 |
| **NDCG@25** | 0.5030 | **NDCG@90** | 0.5238 |
| **NDCG@30** | 0.5085 | **NDCG@100** | 0.5243 |
| **NDCG@35** | 0.5124 | **NDCG@200** | 0.5264 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License 

This project is open-source and available under the [MIT License](https://github.com/mdmohsin212/Visual-Product-Search/blob/main/LICENCE).