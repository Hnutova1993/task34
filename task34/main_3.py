import numpy as np
import nltk
from openai import AsyncOpenAI
import asyncio
import openai
import os
import pandas as pd
import time
from itertools import combinations

data_root_dir = "data/"
output_csv_dir = "output/"

def get_reward(chunks, embedding_dots, sentences_cnt):
    sub_chunks = make_sub_chunks_np(chunks)
    intra_pairs = []
    
    for chunk in sub_chunks:
        for (start_i, len_i), (start_j, len_j) in combinations(chunk, 2):
            idx_i = sentences_cnt * (len_i - 1) - len_i // 3 + start_i
            idx_j = sentences_cnt * (len_j - 1) - len_j // 3 + start_j
            intra_pairs.append((idx_i, idx_j))
    
    intra_indices_i, intra_indices_j = zip(*intra_pairs) if intra_pairs else ([], [])
    intra_indices_i = np.array(intra_indices_i, dtype=np.int32)
    intra_indices_j = np.array(intra_indices_j, dtype=np.int32)
    
    all_indices = []
    for chunk in sub_chunks:
        for start, length in chunk:
            all_indices.append(sentences_cnt * (length - 1) - length // 3 + start)
    all_indices = np.array(all_indices, dtype=np.int32)
    
    x, y = np.meshgrid(all_indices, all_indices)
    m, n = np.meshgrid(np.repeat(np.arange(len(sub_chunks)), [len(c) for c in sub_chunks]), 
                       np.repeat(np.arange(len(sub_chunks)), [len(c) for c in sub_chunks]))
    inter_mask = (m != n)
    inter_indices_i = x[inter_mask]
    inter_indices_j = y[inter_mask]
    
    intra_dot_sum = np.sum(embedding_dots[intra_indices_i, intra_indices_j]) if len(intra_indices_i) > 0 else 0
    inter_dot_sum = np.sum(embedding_dots[inter_indices_i, inter_indices_j]) if len(inter_indices_i) > 0 else 0
    
    intra_dot_cnt = len(intra_indices_i)
    inter_dot_cnt = len(inter_indices_i)
    
    return (intra_dot_sum / intra_dot_cnt if intra_dot_cnt != 0 else 0) - (
        inter_dot_sum / inter_dot_cnt if inter_dot_cnt != 0 else 0
    )

def make_sub_chunks_np(chunks):
    sub_chunks = []
    start_chunk = 0
    for i in range(len(chunks)):
        one_sub_chunks = []
        if i > 0:
            start_chunk = chunks[i-1] + 1
        num_full_chunks = (chunks[i] - start_chunk + 1) // 3
        remainder = (chunks[i] - start_chunk + 1) % 3
        if num_full_chunks > 0:
            starts = np.arange(num_full_chunks) * 3 + start_chunk
            one_sub_chunks.extend([(start, 3) for start in starts])
        if remainder > 0:
            one_sub_chunks.append((start_chunk + num_full_chunks * 3, remainder))
        sub_chunks.append(one_sub_chunks)
    return sub_chunks

def validate_chunk_sizes(chunks, sentences_lengths, chunk_size):
    prev_end = -1
    for end in chunks:
        chunk_len = np.sum(sentences_lengths[prev_end+1:end+1])
        if chunk_len > chunk_size:
            raise ValueError(f"Chunk size exceeded: {chunk_len} > {chunk_size}")
        prev_end = end
    return True

async def _chunking(text, chunk_size, chunk_cnt, file_num):
    sentences = nltk.sent_tokenize(text)
    sentences_cnt = len(sentences)
    sentences_lengths = np.array([len(sent) + 1 for sent in sentences], dtype=np.int32)
    
    # Validate individual sentences
    for i, length in enumerate(sentences_lengths):
        if length > chunk_size:
            raise ValueError(f"Sentence {i} exceeds chunk size: {length} > {chunk_size}")

    # Generate extended sentences
    extended_sentences = sentences.copy()
    for i in range(sentences_cnt - 1):
        extended_sentences.append(" ".join(sentences[i:i+2]))
    for i in range(sentences_cnt - 2):
        extended_sentences.append(" ".join(sentences[i:i+3]))
    
    # Load embeddings
    embeddings_file_npy = f"{file_num}.npy"
    if os.path.exists(embeddings_file_npy):
        embeddings = np.load(embeddings_file_npy, allow_pickle=True)
    else:
        client = AsyncOpenAI(api_key=openai_api_key)
        res = await client.embeddings.create(
            input=extended_sentences,
            model="text-embedding-ada-002",
        )
        embeddings = np.array([item.embedding for item in res.data])
        np.save(embeddings_file_npy, embeddings)
    
    embedding_dots = embeddings @ embeddings.T
    
    # Enhanced initial chunk formation with strict size control
    initial_chunks = []
    cur_pos = 0
    while cur_pos < sentences_cnt:
        max_k = 0
        max_avg_sim = -np.inf
        valid_ks = []
        
        # Find all valid k values first
        for k in range(1, min(5, sentences_cnt - cur_pos) + 1):
            total_length = np.sum(sentences_lengths[cur_pos:cur_pos + k])
            if total_length <= chunk_size:
                valid_ks.append(k)
        
        # Find best k among valid options
        for k in valid_ks:
            sub_matrix = embedding_dots[cur_pos:cur_pos + k, cur_pos:cur_pos + k]
            sum_sim = np.sum(np.triu(sub_matrix, k=1))
            num_pairs = k * (k - 1) // 2
            avg_sim = sum_sim / num_pairs if num_pairs > 0 else 0
            if avg_sim > max_avg_sim or (avg_sim == max_avg_sim and k > max_k):
                max_avg_sim = avg_sim
                max_k = k
                
        # Fallback to largest possible valid k
        if not valid_ks:
            for k in range(min(5, sentences_cnt - cur_pos), 0, -1):
                if np.sum(sentences_lengths[cur_pos:cur_pos + k]) <= chunk_size:
                    max_k = k
                    break
        
        end_pos = cur_pos + max_k - 1
        initial_chunks.append((end_pos, np.sum(sentences_lengths[cur_pos:end_pos+1])))
        cur_pos = end_pos + 1

    # Strict merging with boundary validation
    temperature = 1.0
    cooling_rate = 0.95
    while len(initial_chunks) > chunk_cnt:
        valid_merges = []
        for i in range(len(initial_chunks) - 1):
            merged_length = initial_chunks[i][1] + initial_chunks[i+1][1]
            if merged_length <= chunk_size:
                valid_merges.append(i)
        
        if not valid_merges:
            raise ValueError("Cannot achieve target chunk count without violating size constraints")
        
        # Find best merge
        best_merge = None
        best_reward = -np.inf
        for i in valid_merges:
            merged_end = initial_chunks[i+1][0]
            new_chunks = initial_chunks[:i] + [(merged_end, initial_chunks[i][1] + initial_chunks[i+1][1])] + initial_chunks[i+2:]
            reward = get_reward([c[0] for c in new_chunks], embedding_dots, sentences_cnt)
            if reward > best_reward:
                best_reward = reward
                best_merge = i
        
        # Apply merge with simulated annealing
        current_reward = get_reward([c[0] for c in initial_chunks], embedding_dots, sentences_cnt)
        if best_reward > current_reward or np.exp((best_reward - current_reward) / temperature) > np.random.random():
            merged_end = initial_chunks[best_merge+1][0]
            initial_chunks = initial_chunks[:best_merge] + [
                (merged_end, initial_chunks[best_merge][1] + initial_chunks[best_merge+1][1])
            ] + initial_chunks[best_merge+2:]
            temperature *= cooling_rate
        else:
            break
    
    chunks = [c[0] for c in initial_chunks]
    
    # Enhanced boundary adjustment with strict size checks
    window_size = 3
    max_iterations = 10
    for _ in range(max_iterations):
        improved = False
        for i in range(len(chunks) - 1):
            current_pos = chunks[i]
            prev_end = chunks[i-1] if i > 0 else -1
            next_end = chunks[i+1]
            
            # Calculate valid adjustment window
            start = max(prev_end + 1, current_pos - window_size)
            end = min(next_end - 1, current_pos + window_size)
            
            best_pos = current_pos
            best_reward = get_reward(chunks, embedding_dots, sentences_cnt)
            
            for pos in range(start, end + 1):
                # Calculate chunk lengths with numpy for accuracy
                current_chunk_len = np.sum(sentences_lengths[prev_end+1:pos+1])
                next_chunk_len = np.sum(sentences_lengths[pos+1:next_end+1])
                
                if current_chunk_len > chunk_size or next_chunk_len > chunk_size:
                    continue
                
                new_chunks = chunks.copy()
                new_chunks[i] = pos
                reward = get_reward(new_chunks, embedding_dots, sentences_cnt)
                
                if reward > best_reward:
                    best_reward = reward
                    best_pos = pos
            
            if best_pos != current_pos:
                chunks[i] = best_pos
                improved = True
                
        if not improved:
            break
    
    # Final validation
    validate_chunk_sizes(chunks, sentences_lengths, chunk_size)
    
    # Generate chunk texts
    chunk_text = []
    js = 0
    for end in chunks:
        chunk_text.append(" ".join(sentences[js:end+1]))
        js = end + 1
    
    return chunk_text

async def main():
    files = [
        {"name": 1, "chunk_size": 2000, "num_chunks": 30},
        {"name": 2, "chunk_size": 3000, "num_chunks": 24},
        {"name": 3, "chunk_size": 4000, "num_chunks": 22},
        {"name": 4, "chunk_size": 4000, "num_chunks": 18},
        {"name": 5, "chunk_size": 2000, "num_chunks": 42},
        {"name": 6, "chunk_size": 4000, "num_chunks": 14},
        {"name": 7, "chunk_size": 4000, "num_chunks": 12},
        {"name": 8, "chunk_size": 2000, "num_chunks": 34}
    ]
    
    for file_info in files:
        start_time = time.time()
        try:
            with open(os.path.join(data_root_dir, f"{file_info['name']}.txt"), "r", encoding="utf-8") as f:
                text = f.read()
            
            chunks = await _chunking(text, file_info['chunk_size'], file_info['num_chunks'], file_info['name'])
            df = pd.DataFrame({
                "Chunk Index": range(1, len(chunks)+1),
                "Chunk Content": chunks,
                "Chunk Length": [len(c) for c in chunks]
            })
            os.makedirs(output_csv_dir, exist_ok=True)
            df.to_csv(os.path.join(output_csv_dir, f"re-{file_info['name']}.csv"), index=False)
            
            print(f"✅ Processed file {file_info['name']} in {time.time() - start_time:.2f}s")
            print(f"   Chunks: {len(chunks)} | Max length: {max(len(c) for c in chunks)}")
        
        except Exception as e:
            print(f"❌ Error processing file {file_info['name']}: {str(e)}")

if __name__ == '__main__':
    asyncio.run(main())