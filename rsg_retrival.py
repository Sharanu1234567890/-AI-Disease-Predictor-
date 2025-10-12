class ThreeInOneRetriever:
    def __init__(self, vector_db, keyword_index, reranker_model):
        self.vector_db = vector_db
        self.keyword_index = keyword_index  
        self.reranker = reranker_model
    
    def retrieve(self, query, metadata_filters=None, top_k=10):
        """
        Complete 3-in-1 retrieval:
        1. Hybrid Search
        2. Metadata Filtering  
        3. Re-ranking
        """
        # STEP 1: Hybrid Search (Broad Recall)
        print("🔍 Step 1: Hybrid Search")
        hybrid_results = self.hybrid_search(query, top_k=100)
        
        if not hybrid_results:
            return []
        
        # STEP 2: Metadata Filtering (Domain Focus)
        print("🎯 Step 2: Metadata Filtering")
        if metadata_filters:
            filtered_results = self.apply_metadata_filters(hybrid_results, metadata_filters)
            print(f"   Filtered: {len(hybrid_results)} → {len(filtered_results)} documents")
        else:
            filtered_results = hybrid_results
        
        if not filtered_results:
            return []
        
        # STEP 3: Re-ranking (Precision Ordering)
        print("⚡ Step 3: Re-ranking")
        final_results = self.rerank(query, filtered_results, top_k=top_k)
        
        return final_results
    
    def hybrid_search(self, query, top_k=100):
        """Step 1: Vector + Keyword search"""
        # Vector search
        vector_results = self.vector_db.similarity_search(query, k=top_k)
        
        # Keyword search
        keyword_results = self.keyword_index.search(query, k=top_k)
        
        # Combine using RRF
        combined = self.reciprocal_rank_fusion(vector_results, keyword_results)
        return combined[:top_k]
    
    def apply_metadata_filters(self, documents, filters):
        """Step 2: Filter by metadata"""
        filtered = []
        for doc in documents:
            if self.metadata_matches(doc.metadata, filters):
                filtered.append(doc)
        return filtered
    
    def rerank(self, query, documents, top_k=10):
        """Step 3: Re-rank for perfect ordering"""
        pairs = [(query, doc.text) for doc in documents]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
    
    def reciprocal_rank_fusion(self, list1, list2, k=60):
        """Combine ranked lists intelligently"""
        scores = {}
        
        for rank, doc in enumerate(list1):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + k)
        
        for rank, doc in enumerate(list2):
            scores[doc] = scores.get(doc, 0) + 1 / (rank + k)
        
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    def metadata_matches(self, doc_metadata, filters):
        """Check if document matches all filters"""
        for key, value in filters.items():
            if doc_metadata.get(key) != value:
                return False
        return True
