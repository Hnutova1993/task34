import numpy as np
import nltk
from openai import AsyncOpenAI
import asyncio
import openai
import os
import pandas as pd
import time

def get_reward(chunks, embedding_dots, sentences_cnt):
    sub_chunks = make_sub_chunks_np(chunks)
    num_sub_chunks = sum(len(sc) for sc in sub_chunks)
    if num_sub_chunks < 2:
        return 0
    intra_indices_i = []
    intra_indices_j = []
    chunk_indices = [] 

    chunk_start = 0
    for i, chunk in enumerate(sub_chunks):
        for j1 in range(len(chunk)):
            for j2 in range(j1 + 1, len(chunk)):
                start_i, len_i = chunk[j1]
                start_j, len_j = chunk[j2]
                intra_indices_i.append(sentences_cnt * (len_i - 1) - len_i // 3 + start_i)
                intra_indices_j.append(sentences_cnt * (len_j - 1) - len_j // 3 + start_j)
        chunk_start += len(chunk)
        chunk_indices.extend([i] * len(chunk))  

    intra_indices_i = np.array(intra_indices_i, dtype=np.int32)
    intra_indices_j = np.array(intra_indices_j, dtype=np.int32)

    inter_indices_i = []
    inter_indices_j = []
    chunk_indices = np.array(chunk_indices, dtype=np.int32)
    all_indices = []

    chunk_start = 0
    for i, chunk in enumerate(sub_chunks):
      for start, length in chunk:
        all_indices.append(sentences_cnt * (length - 1) - length // 3 + start)
    all_indices = np.array(all_indices, dtype=np.int32)

    x, y = np.meshgrid(all_indices, all_indices)
    m, n = np.meshgrid(chunk_indices, chunk_indices)
    inter_mask = m != n 
    inter_indices_i = x[inter_mask]
    inter_indices_j = y[inter_mask]

    intra_dot_sum = np.sum(embedding_dots[intra_indices_i, intra_indices_j]) if len(intra_indices_i) >0 else 0
    inter_dot_sum = np.sum(embedding_dots[inter_indices_i, inter_indices_j]) if len(inter_indices_i) >0 else 0

    intra_dot_cnt = len(intra_indices_i)
    inter_dot_cnt = len(inter_indices_i)

    reward = (intra_dot_sum / intra_dot_cnt if intra_dot_cnt != 0 else 0) - (
        inter_dot_sum / inter_dot_cnt if inter_dot_cnt != 0 else 0
    )
    return reward


def make_sub_chunks_np(chunks):
    sub_chunks = []
    start_chunk = 0
    for i in range(len(chunks)):
        one_sub_chunks = []
        if i>0:
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

async def _chunking(text, chunk_size, chunk_cnt,file_num):
    sentences = nltk.sent_tokenize(text)
    sentences_cnt = len(sentences)
    sentences_lengths = np.zeros(sentences_cnt, dtype=np.int32)
    for i in range(sentences_cnt):
        sentences_lengths[i] = len(sentences[i]) + 1

    for i in range(sentences_cnt - 1):
        sent1 = " ".join(sentences[i:i+2])
        sentences.append(sent1)
    for i in range(sentences_cnt - 2):
        sent1 = " ".join(sentences[i:i+3])
        sentences.append(sent1)
    embeddings_file_npy = str(file_num)+".npy"
    if os.path.exists(embeddings_file_npy):
        # Load embeddings from .npy file
        embeddings = np.load(embeddings_file_npy, allow_pickle=True)
        print("Embeddings loaded from .npy file.")
    else:
        # Generate embeddings if the file doesn't exist
        client = AsyncOpenAI(api_key=openai_api_key)
        res = await client.embeddings.create(
            input=sentences,
            model="text-embedding-ada-002",
        )
        embeddings = np.array([np.array(item.embedding) for item in res.data])
        np.save(embeddings_file_npy,embeddings)

    start_time = time.time()

    embedding_dots = embeddings @ embeddings.T

    initial_chunks = []
    cur_pos = 0
    while cur_pos<sentences_cnt:
    
        if cur_pos<sentences_cnt-2 and sentences_lengths[cur_pos]+sentences_lengths[cur_pos+1]+sentences_lengths[cur_pos+2]<=chunk_size:
            initial_chunks.append((cur_pos+2,sentences_lengths[cur_pos]+sentences_lengths[cur_pos+1]+sentences_lengths[cur_pos+2]))
            cur_pos+=3
        elif cur_pos<sentences_cnt-1 and sentences_lengths[cur_pos]+sentences_lengths[cur_pos+1]<=chunk_size:
            initial_chunks.append((cur_pos+1,sentences_lengths[cur_pos]+sentences_lengths[cur_pos+1]))
            cur_pos+=2
        else:
            initial_chunks.append((cur_pos,sentences_lengths[cur_pos]))
            cur_pos+=1

    while len(initial_chunks)>chunk_cnt:
        max_reward = 0
        max_index = (0,0)
        for i in range(len(initial_chunks)-2):
            start_pos = 0 if i == 0 else initial_chunks[i-1][0] + 1
            for j in range(initial_chunks[i][0],initial_chunks[i+1][0]):
                if sentences_lengths[start_pos:j+1].sum() > chunk_size :
                    break
                elif sentences_lengths[j+1:initial_chunks[i+2][0]+1].sum() > chunk_size:
                    continue
                chunk_tmp, _ = zip(*initial_chunks)
                chunk_tmp = list(chunk_tmp)
                chunk_tmp[i] = j
                chunk_tmp.pop(i+1)
                reward = get_reward(chunk_tmp,embedding_dots,sentences_cnt)
                if reward>max_reward:
                    max_reward = reward
                    max_index = (i,j)
        tmp = list(initial_chunks[max_index[0]])
        tmp[0] = max_index[1]
        start_pos = 0 if max_index[0] == 0 else initial_chunks[max_index[0]-1][0] + 1
        tmp[1] = sentences_lengths[start_pos:max_index[1]+1].sum()
        initial_chunks[max_index[0]] = tuple(tmp)
        initial_chunks.pop(max_index[0]+1)
        tmp = list(initial_chunks[max_index[0]+1])
        tmp[1] = sentences_lengths[max_index[1]+1:initial_chunks[max_index[0]+1][0]+1].sum()
        initial_chunks[max_index[0]+1] = tuple(tmp)
    chunks, chunk_lengths = zip(*initial_chunks)
    chunks = list(chunks)
    chunk_lengths = list(chunk_lengths)


    max_reward = get_reward(chunks,embedding_dots,sentences_cnt)
    last_max = 0
    while max_reward - last_max > 1e-4:
        start_current_chunk = 0
        last_max = max_reward
        for cur_chunk_index in range(chunk_cnt-1):
            current_length = chunk_lengths[cur_chunk_index+1]
            start_pos = chunks[cur_chunk_index]
            
            while current_length +sentences_lengths[start_pos] <= chunk_size and start_pos>start_current_chunk:
                current_length += sentences_lengths[start_pos]
                start_pos -= 1
            current_length = chunk_lengths[cur_chunk_index]
            end_pos = chunks[cur_chunk_index]
            while current_length + sentences_lengths[end_pos+1] < chunk_size and end_pos<chunks[cur_chunk_index+1]-1:
                current_length += sentences_lengths[end_pos+1]
                end_pos += 1

            for cur_pos in range(start_pos,end_pos+1):
                chunks_tmp = chunks.copy()
                chunks_tmp[cur_chunk_index] = cur_pos
                
                reward = get_reward(chunks_tmp,embedding_dots,sentences_cnt)
                if(reward>max_reward):
                    max_reward=reward
                    chunks[cur_chunk_index] = cur_pos
                    chunk_lengths[cur_chunk_index] = np.sum(sentences_lengths[start_current_chunk:cur_pos + 1])
                    chunk_lengths[cur_chunk_index + 1] = np.sum(sentences_lengths[cur_pos + 1:chunks[cur_chunk_index + 1] + 1])
            start_current_chunk = chunks[cur_chunk_index] +1
    last_max = 0
    while max_reward - last_max > 1e-4:
        start_current_chunk = 0
        last_max = max_reward
        for cur_chunk_index in range(chunk_cnt - 2):
            start_pos = start_current_chunk
            end_pos = chunks[cur_chunk_index + 2] - 2
            chunks_tmp = chunks.copy()

            for pos1 in range(start_pos, end_pos + 1):
                for pos2 in range(pos1 + 1, end_pos + 2):
                    chunks_tmp[cur_chunk_index] = pos1
                    chunks_tmp[cur_chunk_index + 1] = pos2

                    len1 = sentences_lengths[start_current_chunk : pos1 + 1].sum()
                    if len1 > chunk_size:
                        continue
                    len2 = sentences_lengths[pos1 + 1 : pos2 + 1].sum()
                    if len2 > chunk_size:
                        continue
                    len3 = sentences_lengths[pos2 + 1 : chunks[cur_chunk_index + 2] + 1].sum()
                    if len3 > chunk_size:
                        continue

                    reward = get_reward(chunks_tmp, embedding_dots, sentences_cnt)
                    if reward > max_reward:
                        max_reward = reward
                        chunks = chunks_tmp.copy()
                        chunk_lengths[cur_chunk_index] = len1
                        chunk_lengths[cur_chunk_index + 1] = len2
                        chunk_lengths[cur_chunk_index + 2] = len3
            start_current_chunk = chunks[cur_chunk_index]+1
            
    end_time = time.time()
    print(f"took {end_time - start_time}s to complete calculation")
    print(np.max(chunk_lengths))
    print("------------------------------",max_reward)
    with open("result.txt", "a") as file:
        value1 = file_num  # Example integer
        value2 = max_reward  # Example float
        value3 = end_time - start_time
        file.write(f"{value1}\t{value2}\t{value3}\n")
    js=0
    chunk_text = []
    for i in range(len(chunks)):
        sent1 =" ".join(sentences[js:chunks[i]+1])
        chunk_text.append(sent1)
        js = chunks[i] + 1
    
    return chunk_text 
    
    

async def main():
    index = 1
    # files = [{"name":"1.txt","chunk_size":2000,"num_chunks":32},{"name":"2.txt","chunk_size":4000,"num_chunks":18},{"name":"3.txt","chunk_size":4000,"num_chunks":16},{"name":"4.txt","chunk_size":2000,"num_chunks":32},{"name":"5.txt","chunk_size":3000,"num_chunks":14},{"name":"6.txt","chunk_size":3000,"num_chunks":16}]
    # files = [{"name":1,"chunk_size":2000,"num_chunks":30},{"name":2,"chunk_size":3000,"num_chunks":24},{"name":3,"chunk_size":4000,"num_chunks":22},{"name":4,"chunk_size":4000,"num_chunks":18},{"name":5,"chunk_size":2000,"num_chunks":42},{"name":6,"chunk_size":4000,"num_chunks":14},{"name":7,"chunk_size":4000,"num_chunks":12},{"name":8,"chunk_size":2000,"num_chunks":34}]
    files = [{"name":"test1","chunk_size":4000,"num_chunks":18}]
    for i in range(len(files)):
        start_time = time.time()
        with open(str(files[i]['name'])+".txt","r", encoding="utf-8") as file:
            text = file.read()

        chunks = await _chunking(text, chunk_size=files[i]['chunk_size'], chunk_cnt=files[i]['num_chunks'], file_num=files[i]['name'])
        chunks_file = []
        for j, chunk in enumerate(chunks):
            chunks_file.append({"Chunk Index":j+1,"Chunk Content":chunk})

        df = pd.DataFrame(chunks_file)

        output_file = f"re-{i+1}.csv"
        df.to_csv(output_file, index=False)  

        print("CSV file saved successfully!")
        end_time = time.time()
        index += 1
        print(f"{str(files[i]['name'])+".txt"} : {output_file}")
        print(f"{str(files[i]['name'])+".txt"} took {end_time - start_time}s to complete chunking")

if __name__ == '__main__':
    asyncio.run(main())