# from visual_product_search.indexing.search import DatabaseSearch
# from visual_product_search.embeddings.embed import get_image_embedding, get_text_embedding
# from visual_product_search.logger import logging
# from visual_product_search.exception import ExceptionHandle
# import streamlit as st
# import numpy as np
# from PIL import Image
# import sys

# st.set_page_config(page_title="Visual Product Search", layout="wide")
# st.title("🛍️ Visual Product Search")
# st.write("Upload an image or enter a text description to find similar fashion products.")

# tab1, tab2 = st.tabs(["Image Search", "Text Search"])

# with tab1:
#     uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "png", "jpeg"])
#     if uploaded_file:
#         try:
#             image = Image.open(uploaded_file).convert("RGB")
#             st.image(image, caption="Uploaded Image", use_column_width=True)
            
#             # Get embedding
#             query_vec = get_image_embedding(uploaded_file, model, processor, device)
#             results = indexer.search([query_vec], limit=5)
            
#             st.subheader("Top 5 Similar Products:")
#             for r in results[0]:
#                 st.write(f"Score: {r.score}, Metadata: {r.entity.get('metadata')}")

#         except Exception as e:
#             logging.error("Image search failed")
#             st.error(f"Error: {e}")

# with tab2:
#     text_query = st.text_input("Enter text description (e.g., 'black shoes for men')")
#     if text_query:
#         try:
#             query_vec = get_text_embedding(text_query, model, processor, device)
#             results = indexer.search([query_vec], limit=5)
            
#             st.subheader("Top 5 Similar Products:")
#             for r in results[0]:
#                 st.write(f"Score: {r.score}, Metadata: {r.entity.get('metadata')}")

#         except Exception as e:
#             logging.error("Text search failed")
#             st.error(f"Error: {e}")