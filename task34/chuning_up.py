import numpy as np
import nltk
from openai import AsyncOpenAI
import time
import asyncio

import pandas as pd
import os
import openai

def chunking_update_1(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks0, reps, mode):
    dim_emb = 1536
    sent_qty2 = sent_qty + sent_qty - 1
    INTRA = 0
    INTRA_NO = 1
    INTER = 2
    INTER_NO = 3

    chunks = np.copy(chunks0)
    chunk_lengths = np.zeros(chunk_qty)
    for i in range(chunk_qty):
        chunk_start = 0 if i == 0 else chunks[i-1] + 1
        for j in range(chunk_start, chunks[i]+1):
            chunk_lengths[i] += sent_lengths[j]
    
    subchunk_qtys0 = np.zeros(chunk_qty,dtype= np.int32)
    subchunk_starts0 = np.zeros(chunk_qty, dtype=np.int32)
    subchunk_embnums0 = np.zeros(sent_qty, dtype=np.int32)
    dot = np.zeros(4)
    emb_no = 0
    for i in range(chunk_qty):
        chunk_start = 0 if i == 0 else chunks[i-1] + 1
        subchunk_qtys0[i] = ((chunks[i] -  chunk_start) // 3) + 1
        subchunk_starts0[i] = emb_no
        emb_no += (subchunk_qtys0[i] + 1)
        for j in range(subchunk_qtys0[i]):
            subchunk_embnums0[subchunk_starts0[i]+j] =  sent_qty2 + chunk_start + j * 3
        mod3 = (chunks[i] - chunk_start + 1) % 3
        if mod3 == 1:
            subchunk_embnums0[subchunk_starts0[i]+subchunk_qtys0[i]-1] = chunks[i]
        elif mod3 == 2:
            subchunk_embnums0[subchunk_starts0[i]+subchunk_qtys0[i]-1] = sent_qty + chunks[i] - 1
        
        for j1 in range(subchunk_qtys0[i]-1):
            for j2 in range(j1+1, subchunk_qtys0[i]):
                dot[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[i]+j1], subchunk_embnums0[subchunk_starts0[i]+j2]]
                dot[INTRA_NO] += 1
        for j in range(i):
            for j1 in range(subchunk_qtys0[i]):
                for j2 in range(subchunk_qtys0[j]):
                    dot[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[i]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                    dot[INTER_NO] += 1
    m_inter_dot = dot[INTER] / dot[INTER_NO] if dot[INTER_NO] != 0 else 0
    m_intra_dot = dot[INTRA] / dot[INTRA_NO] if dot[INTRA_NO] != 0 else 0
    reward = m_intra_dot -  m_inter_dot

    subchunk_qtys1 = np.copy(subchunk_qtys0)
    subchunk_qtys2 = np.copy(subchunk_qtys0)
    subchunk_starts1 = np.copy(subchunk_starts0)
    subchunk_starts2 = np.copy(subchunk_starts0)
    subchunk_embnums1 = np.copy(subchunk_embnums0)
    subchunk_embnums2 = np.copy(subchunk_embnums0)
    dot0 = np.copy(dot)
    dot1 = np.copy(dot)
    dot2 = np.copy(dot)

    subchunk_qtys_r = np.zeros(2)
    subchunk_embnums_r = np.zeros(40)

    for rep in range(reps):
        update = 0
        if mode == 1:
            indrange = range(chunk_qty-1)
        elif mode == 2:
            indrange = range(chunk_qty-2,-1,-1)
        elif mode == 3:
            indrange = range(chunk_qty-1) if (rep % 2) == 0 else range(chunk_qty-2, -1, -1)
        elif mode == 4:
            indrange = range(chunk_qty-1) if (rep % 2) == 1 else range(chunk_qty-2, -1, -1)
        
        for ind in indrange:
            ind1 = ind + 1
            chunk_start = 0 if ind == 0 else chunks[ind-1] + 1
            dotm = np.zeros(4)
            for j1 in range(subchunk_qtys0[ind]-1):
                for j2 in range(j1+1, subchunk_qtys0[ind]):
                    dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind]+j2]]
                    dotm[INTRA_NO] += 1
            for j1 in range(subchunk_qtys0[ind1]-1):
                for j2 in range(j1+1, subchunk_qtys0[ind1]):
                    dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                    dotm[INTRA_NO] += 1
            for j1 in range(subchunk_qtys0[ind]):
                for j2 in range(subchunk_qtys0[ind1]):
                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                    dotm[INTER_NO] += 1
            for j in range(chunk_qty):
                if (j != ind) and (j != ind + 1):
                    for j2 in range(subchunk_qtys0[j]):
                        for j1 in range(subchunk_qtys0[ind]):
                            dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                            dotm[INTER_NO] += 1
                        for j1 in range(subchunk_qtys0[ind1]):
                            dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                            dotm[INTER_NO] += 1
            dotb = dot0 - dotm

            subchunk_qtys_r[0] = subchunk_qtys0[ind]
            subchunk_qtys_r[1] = subchunk_qtys0[ind1]
            subchunk_embnums_r[0:subchunk_qtys1[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
            subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
            subchunk_embno_r = subchunk_starts0[ind1]

            ch0_ind = -1
            chunk_length1 = chunk_lengths[ind1]
            for i in range(chunks[ind], chunk_start, -1):
                if chunk_length1 + sent_lengths[i] > chunk_size:
                    ch0_ind = i
                    break
                chunk_length1 = chunk_length1 + sent_lengths[i]
            if ch0_ind == -1:
                ch0_ind = chunk_start

            chunk_length12 = chunk_lengths[ind] + chunk_lengths[ind1]
            chunk_length1 = chunk_length12 - chunk_length1
            for i in range(ch0_ind, chunks[ind1]):
                if i > ch0_ind:
                    if chunk_length1 + sent_lengths[i] > chunk_size:
                        break
                    chunk_length1 = chunk_length1 + sent_lengths[i]
                mod3 = (i - chunk_start + 1) % 3
                if i - ch0_ind < 3:
                    if mod3 == 1:
                        subchunk_qtys1[ind] = ((i- chunk_start) // 3) + 1
                        subchunk_qtys1[ind1] = ((chunks[ind1] - i - 1) // 3) + 1
                        subchunk_cnt = 0
                        for j in range(chunk_start, i-1, 3):
                            subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = sent_qty2 + j
                            subchunk_cnt += 1
                        k = (i - chunk_start + 1) % 3
                        if k == 1:
                            subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = i
                        elif k == 2:
                            subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = sent_qty + i - 1
                        else:
                            subchunk_cnt -= 1
                        subchunk_starts1[ind1] = subchunk_starts1[ind] + subchunk_cnt + 1
                        subchunk_cnt = 0
                        for j in range(i+1, chunks[ind1]-1, 3):
                            subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = sent_qty2 + j
                            subchunk_cnt += 1
                        k = (chunks[ind1] - i) % 3
                        if k == 1:
                            subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = chunks[ind1]
                        elif k == 2:
                            subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = sent_qty + chunks[ind1] - 1
                        dotm = np.zeros(4)
                        for j1 in range(subchunk_qtys1[ind]-1):
                            for j2 in range(j1+1,subchunk_qtys1[ind]):
                                dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys1[ind1]-1):
                            for j2  in range(j1+1, subchunk_qtys1[ind1]):
                                dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1],subchunk_embnums1[subchunk_starts1[ind1]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys1[ind]):
                            for j2 in range(subchunk_qtys1[ind1]):
                                dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind1]+j2]]
                                dotm[INTER_NO] += 1
                        for j in range(chunk_qty):
                            if (j == ind) or (j == ind + 1):
                                continue
                            for j2 in range(subchunk_qtys1[j]):
                                for j1 in range(subchunk_qtys1[ind]):
                                    dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                    dotm[INTER_NO] += 1
                                for j1 in range(subchunk_qtys1[ind1]):
                                    dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                    dotm[INTER_NO] += 1
                        dot1 = dotb + dotm
                        m_inter_dot = dot1[INTER] / dot1[INTER_NO] if dot1[INTER_NO] != 0 else 0
                        m_intra_dot = dot1[INTRA] / dot1[INTRA_NO] if dot1[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if (reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys1[ind]
                            subchunk_qtys_r[1] = subchunk_qtys1[ind1]
                            subchunk_embnums_r[0:subchunk_qtys1[ind]] = subchunk_embnums1[subchunk_starts1[ind]:subchunk_starts1[ind]+subchunk_qtys1[ind]]
                            subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]] = subchunk_embnums1[subchunk_starts1[ind1]:subchunk_starts1[ind1]+subchunk_qtys1[ind1]]
                            subchunk_embno_r = subchunk_starts1[ind1]
                            dot = np.copy(dot1)
                            update = 1
                    elif mod3 == 2:
                        subchunk_qtys2[ind] = ((i - chunk_start) // 3) + 1
                        subchunk_qtys2[ind1] = ((chunks[ind1] - i - 1) // 3) + 1
                        subchunk_cnt = 0
                        for j in range(chunk_start, i-1, 3):
                            subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = sent_qty2 + j
                            subchunk_cnt += 1
                        k = (i - chunk_start + 1) % 3
                        if k == 1:
                            subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = i
                        elif k == 2:
                            subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = sent_qty + i - 1
                        else:
                            subchunk_cnt -= 1
                        subchunk_starts2[ind1] = subchunk_starts2[ind] + subchunk_cnt + 1
                        subchunk_cnt = 0
                        for j in range(i+1, chunks[ind1]-1, 3):
                            subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = sent_qty2 + j
                            subchunk_cnt += 1
                        k = (chunks[ind1] - i) % 3
                        if k == 1:
                            subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = chunks[ind1]
                        elif k == 2:
                            subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = sent_qty + chunks[ind1] - 1
                        dotm = np.zeros(4)
                        for j1 in range(subchunk_qtys2[ind]-1):
                            for j2 in range(j1+1, subchunk_qtys2[ind]):
                                dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1],subchunk_embnums2[subchunk_starts2[ind]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys2[ind1]-1):
                            for j2 in range(j1+1, subchunk_qtys2[ind1]):
                                dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[ind1]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys2[ind]):
                            for j2 in range(subchunk_qtys2[ind1]):
                                dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind1]+j2]]
                                dotm[INTER_NO] += 1
                        for j in range(chunk_qty):
                            if (j == ind) or (j == ind + 1):
                                continue
                            for j2 in range(subchunk_qtys2[j]):
                                for j1 in range(subchunk_qtys2[ind]):
                                    dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                    dotm[INTER_NO] += 1
                                for j1 in range(subchunk_qtys2[ind1]):
                                    dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                    dotm[INTER_NO] += 1
                        dot2 = dotb + dotm
                        m_inter_dot = dot2[INTER] / dot2[INTER_NO] if dot2[INTER_NO] != 0 else 0
                        m_intra_dot = dot2[INTRA] / dot2[INTRA_NO] if dot2[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if (reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys2[ind]
                            subchunk_qtys_r[1] = subchunk_qtys2[ind1]
                            subchunk_embnums_r[0:subchunk_qtys2[ind]] = subchunk_embnums2[subchunk_starts2[ind]:subchunk_starts2[ind]+subchunk_qtys2[ind]]
                            subchunk_embnums_r[subchunk_qtys2[ind]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]] = subchunk_embnums2[subchunk_starts2[ind1]:subchunk_starts2[ind1]+subchunk_qtys2[ind1]]
                            subchunk_embno_r = subchunk_starts2[ind1]
                            dot = np.copy(dot2)
                            update = 1
                    else:
                        subchunk_qtys0[ind] = ((i - chunk_start) // 3) + 1
                        subchunk_qtys0[ind1] = ((chunks[ind1] - i - 1) // 3) + 1
                        subchunk_cnt = 0
                        for j in range(chunk_start, i-1, 3):
                            subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt]= sent_qty2 + j
                            subchunk_cnt += 1
                        k = (i - chunk_start + 1) % 3
                        if k == 1:
                            subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt] = i
                        elif k == 2:
                            subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt] = sent_qty + i - 1
                        else:
                            subchunk_cnt -= 1
                        subchunk_starts0[ind1] = subchunk_starts0[ind] + subchunk_cnt + 1
                        subchunk_cnt = 0
                        for j in range(i+1, chunks[ind1]-1, 3):
                            subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = sent_qty2 + j
                            subchunk_cnt += 1
                        k = (chunks[ind1] - i) % 3
                        if k == 1:
                            subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = chunks[ind1]
                        elif k == 2:
                            subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = sent_qty + chunks[ind1] - 1
                        dotm = np.zeros(4)
                        for j1 in range(subchunk_qtys0[ind]-1):
                            for j2 in range(j1+1, subchunk_qtys0[ind]):
                                dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys0[ind1]-1):
                            for j2 in range(j1+1, subchunk_qtys0[ind1]):
                                dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1],subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                                dotm[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys0[ind]):
                            for j2 in range(subchunk_qtys0[ind1]):
                                dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1],subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                                dotm[INTER_NO] += 1
                        for j in range(chunk_qty):
                            if (j == ind) or (j == ind + 1):
                                continue
                            for j2 in range(subchunk_qtys0[j]):
                                for j1 in range(subchunk_qtys0[ind]):
                                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1],subchunk_embnums0[subchunk_starts0[j]+j2]]
                                    dotm[INTER_NO] += 1
                                for j1 in range(subchunk_qtys0[ind1]):
                                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                    dotm[INTER_NO] += 1
                        dot0 = dotb + dotm
                        m_inter_dot = dot0[INTER] / dot0[INTER_NO] if dot0[INTER_NO] != 0 else 0
                        m_intra_dot = dot0[INTRA] / dot0[INTRA_NO] if dot0[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if(reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys0[ind]
                            subchunk_qtys_r[1] = subchunk_qtys0[ind1]
                            subchunk_embnums_r[0:subchunk_qtys0[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
                            subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
                            subchunk_embno_r = subchunk_starts0[ind1]
                            dot = np.copy(dot0)
                            update = 1
                else:
                    if mod3 == 1:
                        dotm = np.zeros(4)
                        for j1 in range(subchunk_qtys1[ind]-1):
                            dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1]]
                            dotm[INTRA_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind1]]]
                            dotm[INTER_NO] += 1
                        for j1 in range(1, subchunk_qtys1[ind1]):
                            dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[ind1]]]
                            dotm[INTRA_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1]]
                            dotm[INTER_NO] += 1
                        for j in range(chunk_qty):
                            if (j == ind) or (j == ind + 1):
                                continue
                            for j2 in range(subchunk_qtys1[j]):
                                dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                dotm[INTER_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                dotm[INTER_NO] += 1
                        dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]], subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1]]
                        dotm[INTER_NO] += 1
                        subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1] = sent_qty2 + i - 3
                        subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]] = i
                        subchunk_qtys1[ind] = subchunk_qtys1[ind] + 1
                        subchunk_qtys1[ind1] = subchunk_qtys1[ind1] - 1
                        subchunk_starts1[ind1] = subchunk_starts1[ind1] + 1
                        dotp = np.zeros(4)
                        dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1], subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-2]]
                        dotp[INTRA_NO] += 1 #####################################################################
                        for j1 in range(subchunk_qtys1[ind]-2):
                            dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1],subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1]]
                            dotp[INTRA_NO] += 1
                            dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1],subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-2]]
                            dotp[INTRA_NO] += 1
                        for j in range(chunk_qty):
                            if j == ind:
                                continue
                            for j2 in range(subchunk_qtys1[j]):
                                dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-1],subchunk_embnums1[subchunk_starts1[j]+j2]]
                                dotp[INTER_NO] += 1
                                dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+subchunk_qtys1[ind]-2],subchunk_embnums1[subchunk_starts1[j]+j2]]
                                dotp[INTER_NO] += 1
                        dot1 = dot1 - dotm + dotp
                        m_inter_dot = dot1[INTER] / dot1[INTER_NO] if dot1[INTER_NO] != 0 else 0
                        m_intra_dot = dot1[INTRA] / dot1[INTRA_NO] if dot1[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if (reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys1[ind]
                            subchunk_qtys_r[1] = subchunk_qtys1[ind1]
                            subchunk_embnums_r[0:subchunk_qtys1[ind]] = subchunk_embnums1[subchunk_starts1[ind]:subchunk_starts1[ind]+subchunk_qtys1[ind]]
                            subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]] = subchunk_embnums1[subchunk_starts1[ind1]:subchunk_starts1[ind1]+subchunk_qtys1[ind1]]
                            subchunk_embno_r = subchunk_starts1[ind1]
                            dot = np.copy(dot1)
                            update = 1
                    elif mod3 == 2:
                        dotm = np.zeros(4)
                        for j1 in range(subchunk_qtys2[ind]-1):
                            dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1]]
                            dotm[INTRA_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind1]]]
                            dotm[INTER_NO] += 1
                        for j1 in range(1, subchunk_qtys2[ind1]):
                            dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[ind1]]]
                            dotm[INTRA_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1]]
                            dotm[INTER_NO] += 1
                        for j in range(chunk_qty):
                            if (j == ind) or (j == ind + 1):
                                continue
                            for j2 in range(subchunk_qtys2[j]):
                                dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                dotm[INTER_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                dotm[INTER_NO] += 1
                        dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1]]
                        dotm[INTER_NO] += 1
                        subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1] = sent_qty2 + i - 4
                        subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]] = sent_qty + i - 1
                        subchunk_qtys2[ind] += 1
                        subchunk_qtys2[ind1] -= 1
                        subchunk_starts2[ind1] += 1
                        dotp = np.zeros(4)
                        dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-2]]
                        dotp[INTRA_NO] += 1
                        for j1 in range(subchunk_qtys2[ind]-2):
                            dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1]]
                            dotp[INTRA_NO] += 1
                            dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-2]]
                            dotp[INTRA_NO] += 1
                        for j in range(chunk_qty):
                            if j == ind:
                                continue
                            for j2 in range(subchunk_qtys2[j]):
                                dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                dotp[INTER_NO] += 1
                                dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+subchunk_qtys2[ind]-2], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                dotp[INTER_NO] += 1
                        dot2 = dot2 - dotm + dotp
                        m_inter_dot = dot2[INTER] / dot2[INTER_NO] if dot2[INTER_NO] != 0 else 0
                        m_intra_dot = dot2[INTRA] / dot2[INTRA_NO] if dot2[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if (reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys2[ind]
                            subchunk_qtys_r[1] = subchunk_qtys2[ind1]
                            subchunk_embnums_r[0:subchunk_qtys2[ind]] = subchunk_embnums2[subchunk_starts2[ind]:subchunk_starts2[ind]+subchunk_qtys2[ind]]
                            subchunk_embnums_r[subchunk_qtys2[ind]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]] = subchunk_embnums2[subchunk_starts2[ind1]:subchunk_starts2[ind1]+subchunk_qtys2[ind1]]
                            subchunk_embno_r = subchunk_starts2[ind1]
                            dot = np.copy(dot2)
                            update = 1
                    else:
                        dotm = np.zeros(4)
                        for j1 in range(1, subchunk_qtys0[ind1]):
                            dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind1]]]
                            dotm[INTRA_NO] += 1
                        for j in range(chunk_qty):
                            if j == ind + 1:
                                continue
                            for j2 in range(subchunk_qtys0[j]):
                                dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                dotm[INTER_NO] += 1
                        subchunk_qtys0[ind] += 1
                        subchunk_qtys0[ind1] -= 1
                        subchunk_starts0[ind1] += 1
                        dotp = np.zeros(4)
                        for j1 in range(subchunk_qtys0[ind]-1):
                            dotp[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind]+subchunk_qtys0[ind]-1]]
                            dotp[INTRA_NO] += 1
                        for j in range(chunk_qty):
                            if j == ind:
                                continue
                            for j2 in range(subchunk_qtys0[j]):
                                dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+subchunk_qtys0[ind]-1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                dotp[INTER_NO] += 1
                        dot0 = dot0 - dotm + dotp
                        m_inter_dot = dot0[INTER] / dot0[INTER_NO] if dot0[INTER_NO] != 0 else 0
                        m_intra_dot = dot0[INTRA] / dot0[INTRA_NO] if dot0[INTRA_NO] != 0 else 0
                        reward2 = m_intra_dot - m_inter_dot
                        if (reward2 > reward) and (chunks[ind] != i):
                            chunks[ind] = i
                            chunk_lengths[ind] = chunk_length1
                            chunk_lengths[ind1] = chunk_length12 - chunk_length1
                            reward = reward2
                            subchunk_qtys_r[0] = subchunk_qtys0[ind]
                            subchunk_qtys_r[1] = subchunk_qtys0[ind1]
                            subchunk_embnums_r[0:subchunk_qtys0[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
                            subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
                            subchunk_embno_r = subchunk_starts0[ind1]
                            dot = np.copy(dot0)
                            update = 1


            subchunk_qtys0[ind] = subchunk_qtys_r[0]
            subchunk_qtys0[ind1] = subchunk_qtys_r[1]
            subchunk_starts0[ind1] = subchunk_embno_r
            subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]] = subchunk_embnums_r[0:subchunk_qtys0[ind]]
            subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]] = subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]]
            subchunk_qtys1[ind] = subchunk_qtys_r[0]
            subchunk_qtys1[ind1] = subchunk_qtys_r[1]
            subchunk_starts1[ind1] = subchunk_embno_r
            subchunk_embnums1[subchunk_starts1[ind]:subchunk_starts1[ind]+subchunk_qtys1[ind]] = subchunk_embnums_r[0:subchunk_qtys1[ind]]
            subchunk_embnums1[subchunk_starts1[ind1]:subchunk_starts1[ind1]+subchunk_qtys1[ind1]] = subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]]
            subchunk_qtys2[ind] = subchunk_qtys_r[0]
            subchunk_qtys2[ind1] = subchunk_qtys_r[1]
            subchunk_starts2[ind1] = subchunk_embno_r
            subchunk_embnums2[subchunk_starts2[ind]:subchunk_starts2[ind]+subchunk_qtys2[ind]] = subchunk_embnums_r[0:subchunk_qtys2[ind]]
            subchunk_embnums2[subchunk_starts2[ind1]:subchunk_starts2[ind1]+subchunk_qtys2[ind1]] = subchunk_embnums_r[subchunk_qtys2[ind]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]]
            dot0 = np.copy(dot)
        
        if update == 0:
            break
    result = {}
    result[0] = chunks
    result[1] = reward
    result[2] = update
    return result

def chunking_update_2(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks0, reps, mode):
    sent_qty2 = sent_qty + sent_qty - 1
    INTRA = 0
    INTRA_NO = 1
    INTER = 2
    INTER_NO = 3
    chunks = np.copy(chunks0)
    chunk_lengths = np.zeros(chunk_qty)
    for i in range(chunk_qty):
        chunk_start = 0 if i == 0 else chunks[i-1] + 1
        for j in range(chunk_start, chunks[i]+1):
            chunk_lengths[i] += sent_lengths[j]

    subchunk_qtys0 = np.zeros(chunk_qty, dtype=np.int32)
    subchunk_starts0 = np.zeros(chunk_qty, dtype=np.int32)
    subchunk_embnums0 = np.zeros(sent_qty, dtype=np.int32)
    dot = np.zeros(4)
    emb_no = 0
    for i in range(chunk_qty):
        chunk_start = 0 if i == 0 else chunks[i-1] + 1
        subchunk_qtys0[i] = (chunks[i] - chunk_start ) // 3 + 1
        subchunk_starts0[i] = emb_no
        emb_no += (subchunk_qtys0[i] + 1)
        for j in range(subchunk_qtys0[i]):
            subchunk_embnums0[subchunk_starts0[i]+j] = sent_qty2 + chunk_start + j * 3
        mod3 = (chunks[i] - chunk_start + 1) % 3
        if mod3 == 1:
            subchunk_embnums0[subchunk_starts0[i]+subchunk_qtys0[i]-1] = chunks[i]
        elif mod3 == 2:
            subchunk_embnums0[subchunk_starts0[i]+subchunk_qtys0[i]-1] = sent_qty + chunks[i] - 1

        for j1 in range(subchunk_qtys0[i]-1):
            for j2 in range(j1+1, subchunk_qtys0[i]):
                dot[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[i]+j1], subchunk_embnums0[subchunk_starts0[i]+j2]]
                dot[INTRA_NO] += 1
        for j in range(i):
            for j1 in range(subchunk_qtys0[i]):
                for j2 in range(subchunk_qtys0[j]):
                    dot[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[i]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                    dot[INTER_NO] += 1
    m_intra_dot = dot[INTRA] / dot[INTRA_NO] if dot[INTRA_NO] != 0 else 0
    m_inter_dot = dot[INTER] / dot[INTER_NO] if dot[INTER_NO] != 0 else 0
    reward = m_intra_dot - m_inter_dot

    subchunk_qtys1 = np.copy(subchunk_qtys0)
    subchunk_qtys2 = np.copy(subchunk_qtys0)
    subchunk_starts1 = np.copy(subchunk_starts0)
    subchunk_embnums1 = np.copy(subchunk_embnums0)
    dot0 = np.copy(dot)
    subchunk_qtys_r = np.zeros(3)
    subchunk_starts_r = np.zeros(2)
    subchunk_embnums_r = np.zeros(40)

    for rep in range(reps):
        update = 0
        if mode == 1:
            indrange = range(chunk_qty - 2)
        elif mode == 2:
            indrange = range(chunk_qty - 3, -1, -1)
        elif mode == 3:
            indrange = range(chunk_qty - 2) if (rep % 2) == 0 else range(chunk_qty - 3, -1, -1)
        elif mode == 4:
            indrange = range(chunk_qty - 2) if (rep % 2) == 1 else range(chunk_qty - 3, -1, -1)
        for ind in indrange:
            ind1 = ind + 1
            ind2 = ind + 2
            emb_no = 1
            for i in range(chunk_qty):
                subchunk_starts1[i] = emb_no
                emb_no += (subchunk_qtys1[i] + 1)
                subchunk_embnums1[subchunk_starts1[i]+subchunk_qtys1[i]] = 0
                for j in range(subchunk_qtys1[i]):
                    subchunk_embnums1[subchunk_starts1[i]+j] = subchunk_embnums0[subchunk_starts0[i]+j]
            subchunk_starts0 = np.copy(subchunk_starts1)
            subchunk_starts2 = np.copy(subchunk_starts1)
            subchunk_embnums0 = np.copy(subchunk_embnums1)
            subchunk_embnums2 = np.copy(subchunk_embnums1)

            chunk_start = 0 if ind == 0 else chunks[ind-1] + 1
            dotm = np.zeros(4)
            for j1 in range(subchunk_qtys0[ind]-1):
                for j2 in range(j1+1, subchunk_qtys0[ind]):
                    dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind]+j2]]
                    dotm[INTRA_NO] += 1
            for j1 in range(subchunk_qtys0[ind1]-1):
                for j2 in range(j1+1, subchunk_qtys0[ind1]):
                    dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                    dotm[INTRA_NO] += 1
            for j1 in range(subchunk_qtys0[ind2] - 1):
                for j2 in range(j1+1, subchunk_qtys0[ind2]):
                    dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                    dotm[INTRA_NO] += 1
            for j1 in range(subchunk_qtys0[ind]):
                for j2 in range(subchunk_qtys0[ind1]):
                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                    dotm[INTER_NO] += 1
            for j1 in range(subchunk_qtys0[ind]):
                for j2 in range(subchunk_qtys0[ind2]):
                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                    dotm[INTER_NO] += 1
            for j1 in range(subchunk_qtys0[ind1]):
                for j2 in range(subchunk_qtys0[ind2]):
                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                    dotm[INTER_NO] += 1
            for j in range(chunk_qty):
                if (j == ind) or (j == ind1) or (j == ind2):
                    continue
                for j2 in range(subchunk_qtys0[j]):
                    for j1 in range(subchunk_qtys0[ind]):
                        dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                        dotm[INTER_NO] += 1
                    for j1 in range(subchunk_qtys0[ind1]):
                        dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                        dotm[INTER_NO] += 1
                    for j1 in range(subchunk_qtys0[ind2]):
                        dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                        dotm[INTER_NO] += 1
            dotb = dot0 - dotm

            subchunk_qtys_r[0] = subchunk_qtys0[ind]
            subchunk_qtys_r[1] = subchunk_qtys0[ind1]
            subchunk_qtys_r[2] = subchunk_qtys0[ind2]
            subchunk_embnums_r[0:subchunk_qtys0[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
            subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
            subchunk_embnums_r[subchunk_qtys0[ind]+subchunk_qtys0[ind1]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]+subchunk_qtys0[ind2]] = subchunk_embnums0[subchunk_starts0[ind2]:subchunk_starts0[ind2]+subchunk_qtys0[ind2]]
            subchunk_starts_r[0] = subchunk_starts0[ind1]
            subchunk_starts_r[1] = subchunk_starts0[ind2]

            chunk_length123 = chunk_lengths[ind] + chunk_lengths[ind1] + chunk_lengths[ind2]
            chunk_length1 = 0
            for i1 in range(chunk_start,chunks[ind2]-1):
                chunk_length1 += sent_lengths[i1]
                if chunk_length1 > chunk_size:
                    break
                cnt = 0
                chunk_length2 = 0
                for i2 in range(i1+1,chunks[ind2]):
                    chunk_length2 += sent_lengths[i2]
                    if chunk_length2 > chunk_size:
                        break
                    chunk_length3 = chunk_length123 - chunk_length1 - chunk_length2
                    if chunk_length3 > chunk_size:
                        continue
                    cnt += 1
                    mod3 = (i2 - i1) % 3
                    if cnt <= 3:
                        if mod3 == 1:
                            subchunk_qtys1[ind] = ((i1 -chunk_start) // 3) + 1
                            subchunk_qtys1[ind1] = ((i2 - i1 - 1) // 3) + 1
                            subchunk_qtys1[ind2] = ((chunks[ind2] - i2 - 1) // 3) + 1
                            subchunk_cnt = 0
                            for j in range(chunk_start, i1 - 1, 3):
                                subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i1 - chunk_start + 1) % 3
                            if k == 1:
                                subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = i1
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums1[subchunk_starts1[ind]+subchunk_cnt] = sent_qty + i1 - 1
                                subchunk_cnt += 1
                            subchunk_starts1[ind1] = subchunk_starts1[ind] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i1+1, i2 - 1, 3):
                                subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i2 - i1) % 3
                            if k == 1:
                                subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = i2
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums1[subchunk_starts1[ind1]+subchunk_cnt] = sent_qty + i2 - 1
                                subchunk_cnt += 1
                            subchunk_starts1[ind2] = subchunk_starts1[ind1] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i2+1, chunks[ind2] - 1, 3):
                                subchunk_embnums1[subchunk_starts1[ind2]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (chunks[ind2] - i2) % 3
                            if k == 1:
                                subchunk_embnums1[subchunk_starts1[ind2]+subchunk_cnt] = chunks[ind2]
                            elif k == 2:
                                subchunk_embnums1[subchunk_starts1[ind2]+subchunk_cnt] = sent_qty + chunks[ind2] - 1
                            dotp = np.zeros(4)
                            for j1 in range(subchunk_qtys1[ind] - 1):
                                for j2 in range(j1+1, subchunk_qtys1[ind]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys1[ind1]-1):
                                for j2 in range(j1+1, subchunk_qtys1[ind1]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[ind1]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys1[ind2]-1):
                                for j2 in range(j1+1, subchunk_qtys1[ind2]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2]+j1], subchunk_embnums1[subchunk_starts1[ind2]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys1[ind]):
                                for j2 in range(subchunk_qtys1[ind1]):
                                    dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind1]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys1[ind]):
                                for j2 in range(subchunk_qtys1[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys1[ind1]):
                                for j2 in range(subchunk_qtys1[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j in range(chunk_qty):
                                if (j == ind) or (j == ind1) or (j == ind2):
                                    continue
                                for j2 in range(subchunk_qtys1[j]):
                                    for j1 in range(subchunk_qtys1[ind]):
                                        dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind]+j1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys1[ind1]):
                                        dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1]+j1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys1[ind2]):
                                        dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2]+j1], subchunk_embnums1[subchunk_starts1[j]+j2]]
                                        dotp[INTER_NO] += 1
                            dot1 = dotb + dotp
                            m_intra_dot = dot1[INTRA] / dot1[INTRA_NO] if dot1[INTRA_NO] != 0 else 0
                            m_inter_dot = dot1[INTER] / dot1[INTER_NO] if dot1[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys1[ind]
                                subchunk_qtys_r[1] = subchunk_qtys1[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys1[ind2]
                                subchunk_embnums_r[0:subchunk_qtys1[ind]] = subchunk_embnums1[subchunk_starts1[ind]:subchunk_starts1[ind]+subchunk_qtys1[ind]]
                                subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]] = subchunk_embnums1[subchunk_starts1[ind1]:subchunk_starts1[ind1]+subchunk_qtys1[ind1]]
                                subchunk_embnums_r[subchunk_qtys1[ind]+subchunk_qtys1[ind1]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]+subchunk_qtys1[ind2]] = subchunk_embnums1[subchunk_starts1[ind2]:subchunk_starts1[ind2]+subchunk_qtys1[ind2]]
                                subchunk_starts_r[0] = subchunk_starts1[ind1]
                                subchunk_starts_r[1] = subchunk_starts1[ind2]
                                dot0 = np.copy(dot1)
                                update = 1
                        elif mod3 == 2:
                            subchunk_qtys2[ind] = ((i1 - chunk_start) // 3) + 1
                            subchunk_qtys2[ind1] = ((i2 - i1 - 1) // 3) + 1
                            subchunk_qtys2[ind2] = ((chunks[ind2] - i2 - 1) // 3) + 1
                            subchunk_cnt = 0
                            for j in range(chunk_start, i1 - 1, 3):
                                subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i1 - chunk_start + 1) % 3
                            if k == 1:
                                subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = i1
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums2[subchunk_starts2[ind]+subchunk_cnt] = sent_qty + i1 - 1
                                subchunk_cnt += 1
                            subchunk_starts2[ind1] = subchunk_starts2[ind] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i1+1, i2 - 1, 3):
                                subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i2 - i1) % 3
                            if k == 1:
                                subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = i2
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums2[subchunk_starts2[ind1]+subchunk_cnt] = sent_qty + i2 - 1
                                subchunk_cnt += 1
                            subchunk_starts2[ind2] = subchunk_starts2[ind1] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i2+1, chunks[ind2] - 1, 3):
                                subchunk_embnums2[subchunk_starts2[ind2]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (chunks[ind2] - i2) % 3
                            if k == 1:
                                subchunk_embnums2[subchunk_starts2[ind2]+subchunk_cnt] = chunks[ind2]
                            elif k == 2:
                                subchunk_embnums2[subchunk_starts2[ind2]+subchunk_cnt] = sent_qty + chunks[ind2] - 1
                            dotp = np.zeros(4)
                            for j1 in range(subchunk_qtys2[ind] - 1):
                                for j2 in range(j1+1, subchunk_qtys2[ind]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys2[ind1] - 1):
                                for j2 in range(j1+1, subchunk_qtys2[ind1]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[ind1]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys2[ind2] - 1):
                                for j2 in range(j1+1, subchunk_qtys2[ind2]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2]+j1], subchunk_embnums2[subchunk_starts2[ind2]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys2[ind]):
                                for j2 in range(subchunk_qtys2[ind1]):
                                    dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind1]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys2[ind]):
                                for j2 in range(subchunk_qtys2[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys2[ind1]):
                                for j2 in range(subchunk_qtys2[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j in range(chunk_qty):
                                if (j == ind) or (j == ind1) or (j == ind2):
                                    continue
                                for j2 in range(subchunk_qtys2[j]):
                                    for j1 in range(subchunk_qtys2[ind]):
                                        dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind]+j1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys2[ind1]):
                                        dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1]+j1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys2[ind2]):
                                        dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2]+j1], subchunk_embnums2[subchunk_starts2[j]+j2]]
                                        dotp[INTER_NO] += 1
                            dot2 = dotb + dotp
                            m_intra_dot = dot2[INTRA] / dot2[INTRA_NO] if dot2[INTRA_NO] != 0 else 0
                            m_inter_dot = dot2[INTER] / dot2[INTER_NO] if dot2[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys2[ind]
                                subchunk_qtys_r[1] = subchunk_qtys2[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys2[ind2]
                                subchunk_embnums_r[0:subchunk_qtys2[ind]] = subchunk_embnums2[subchunk_starts2[ind]:subchunk_starts2[ind]+subchunk_qtys2[ind]]
                                subchunk_embnums_r[subchunk_qtys2[ind]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]] = subchunk_embnums2[subchunk_starts2[ind1]:subchunk_starts2[ind1]+subchunk_qtys2[ind1]]
                                subchunk_embnums_r[subchunk_qtys2[ind]+subchunk_qtys2[ind1]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]+subchunk_qtys2[ind2]] = subchunk_embnums2[subchunk_starts2[ind2]:subchunk_starts2[ind2]+subchunk_qtys2[ind2]]
                                subchunk_starts_r[0] = subchunk_starts2[ind1]
                                subchunk_starts_r[1] = subchunk_starts2[ind2]
                                dot = np.copy(dot2)
                                update = 1
                        else:
                            subchunk_qtys0[ind] = ((i1 - chunk_start) // 3) + 1
                            subchunk_qtys0[ind1] = ((i2 -i1 - 1) // 3) + 1
                            subchunk_qtys0[ind2] = ((chunks[ind2] - i2 - 1) // 3) + 1
                            subchunk_cnt = 0
                            for j in range(chunk_start, i1 - 1, 3):
                                subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i1 - chunk_start + 1) % 3
                            if k == 1:
                                subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt] = i1
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums0[subchunk_starts0[ind]+subchunk_cnt] = sent_qty + i1 - 1
                                subchunk_cnt += 1
                            subchunk_starts0[ind1] = subchunk_starts0[ind] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i1+1, i2 - 1, 3):
                                subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (i2 - i1) % 3
                            if k == 1:
                                subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = i2
                                subchunk_cnt += 1
                            elif k == 2:
                                subchunk_embnums0[subchunk_starts0[ind1]+subchunk_cnt] = sent_qty + i2 - 1
                                subchunk_cnt += 1
                            subchunk_starts0[ind2] = subchunk_starts0[ind1] + subchunk_cnt
                            subchunk_cnt = 0
                            for j in range(i2+1, chunks[ind2] - 1, 3):
                                subchunk_embnums0[subchunk_starts0[ind2]+subchunk_cnt] = sent_qty2 + j
                                subchunk_cnt += 1
                            k = (chunks[ind2] - i2) % 3
                            if k == 1:
                                subchunk_embnums0[subchunk_starts0[ind2]+subchunk_cnt] = chunks[ind2]
                            elif k == 2:
                                subchunk_embnums0[subchunk_starts0[ind2]+subchunk_cnt] = sent_qty + chunks[ind2] - 1
                            dotp = np.zeros(4)
                            for j1 in range(subchunk_qtys0[ind] - 1):
                                for j2 in range(j1+1, subchunk_qtys0[ind]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys0[ind1] - 1):
                                for j2 in range(j1+1, subchunk_qtys0[ind1]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys0[ind2] - 1):
                                for j2 in range(j1+1, subchunk_qtys0[ind2]):
                                    dotp[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                                    dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys0[ind]):
                                for j2 in range(subchunk_qtys0[ind1]):
                                    dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind1]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys0[ind]):
                                for j2 in range(subchunk_qtys0[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j1 in range(subchunk_qtys0[ind1]):
                                for j2 in range(subchunk_qtys0[ind2]):
                                    dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[ind2]+j2]]
                                    dotp[INTER_NO] += 1
                            for j in range(chunk_qty):
                                if (j == ind) or (j == ind1) or (j == ind2):
                                    continue
                                for j2 in range(subchunk_qtys0[j]):
                                    for j1 in range(subchunk_qtys0[ind]):
                                        dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys0[ind1]):
                                        dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                        dotp[INTER_NO] += 1
                                    for j1 in range(subchunk_qtys0[ind2]):
                                        dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2]+j1], subchunk_embnums0[subchunk_starts0[j]+j2]]
                                        dotp[INTER_NO] += 1
                            dot0 = dotb + dotp
                            m_intra_dot = dot0[INTRA] / dot0[INTRA_NO] if dot0[INTRA_NO] != 0 else 0
                            m_inter_dot = dot0[INTER] / dot0[INTER_NO] if dot0[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys0[ind]
                                subchunk_qtys_r[1] = subchunk_qtys0[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys0[ind2]
                                subchunk_embnums_r[0:subchunk_qtys0[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
                                subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
                                subchunk_embnums_r[subchunk_qtys0[ind]+subchunk_qtys0[ind1]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]+subchunk_qtys0[ind2]] = subchunk_embnums0[subchunk_starts0[ind2]:subchunk_starts0[ind2]+subchunk_qtys0[ind2]]
                                subchunk_starts_r[0] = subchunk_starts0[ind1]
                                subchunk_starts_r[1] = subchunk_starts0[ind2]
                                dot = np.copy(dot0)
                                update = 1
                    else:
                        if mod3 == 1:
                            dotm = np.zeros(4)
                            for j1 in range(subchunk_qtys1[ind1] - 1):
                                dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + j1],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1]]
                                dotm[INTRA_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + j1],subchunk_embnums1[subchunk_starts1[ind2]]]
                                dotm[INTER_NO] += 1
                            for j1 in range(1, subchunk_qtys1[ind2]):
                                dotm[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2] + j1],subchunk_embnums1[subchunk_starts1[ind2]]]
                                dotm[INTRA_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2] + j1],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1]]
                                dotm[INTER_NO] += 1
                            for j in range(chunk_qty):
                                if (j == ind1) or (j == ind2):
                                    continue
                                for j2 in range(subchunk_qtys1[j]):
                                    dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1],subchunk_embnums1[subchunk_starts1[j] + j2]]
                                    dotm[INTER_NO] += 1
                                    dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2]],subchunk_embnums1[subchunk_starts1[j] + j2]]
                                    dotm[INTER_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind2]],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1]]
                            dotm[INTER_NO] += 1
                            subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1] = sent_qty2 + i2 - 3
                            subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1]] = i2
                            subchunk_qtys1[ind1] += 1
                            subchunk_qtys1[ind2] -= 1
                            subchunk_starts1[ind2] += 1
                            dotp = np.zeros(4)
                            dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1]-2]]
                            dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys1[ind1] - 2):
                                dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + j1],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1]-1]]
                                dotp[INTRA_NO] += 1
                                dotp[INTRA] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + j1],subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1]-2]]
                                dotp[INTRA_NO] += 1
                            for j in range(chunk_qty):
                                if j == ind1:
                                    continue
                                for j2 in range(subchunk_qtys1[j]):
                                    dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 1],subchunk_embnums1[subchunk_starts1[j] + j2]]
                                    dotp[INTER_NO] += 1
                                    dotp[INTER] += sent_dots[subchunk_embnums1[subchunk_starts1[ind1] + subchunk_qtys1[ind1] - 2],subchunk_embnums1[subchunk_starts1[j] + j2]]
                                    dotp[INTER_NO] += 1
                            dot1 = dot1 - dotm + dotp
                            m_intra_dot = dot1[INTRA] / dot1[INTRA_NO] if dot1[INTRA_NO] != 0 else 0
                            m_inter_dot = dot1[INTER] / dot1[INTER_NO] if dot1[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys1[ind]
                                subchunk_qtys_r[1] = subchunk_qtys1[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys1[ind2]
                                subchunk_embnums_r[0:subchunk_qtys1[ind]] = subchunk_embnums1[subchunk_starts1[ind]:subchunk_starts1[ind]+subchunk_qtys1[ind]]
                                subchunk_embnums_r[subchunk_qtys1[ind]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]] = subchunk_embnums1[subchunk_starts1[ind1]:subchunk_starts1[ind1]+subchunk_qtys1[ind1]]
                                subchunk_embnums_r[subchunk_qtys1[ind]+subchunk_qtys1[ind1]:subchunk_qtys1[ind]+subchunk_qtys1[ind1]+subchunk_qtys1[ind2]] = subchunk_embnums1[subchunk_starts1[ind2]:subchunk_starts1[ind2]+subchunk_qtys1[ind2]]
                                subchunk_starts_r[0] = subchunk_starts1[ind1]
                                subchunk_starts_r[1] = subchunk_starts1[ind2]
                                dot = np.copy(dot1)
                                update = 1
                        elif mod3 == 2:
                            dotm = np.zeros(4)
                            for j1 in range(subchunk_qtys2[ind1] - 1):
                                dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + j1],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1]]
                                dotm[INTRA_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + j1],subchunk_embnums2[subchunk_starts2[ind2]]]
                                dotm[INTER_NO] += 1
                            for j1 in range(1, subchunk_qtys2[ind2]):
                                dotm[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2] + j1],subchunk_embnums2[subchunk_starts2[ind2]]]
                                dotm[INTRA_NO] += 1
                                dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2] + j1],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1]]
                                dotm[INTER_NO] += 1
                            for j in range(chunk_qty):
                                if (j == ind1) or (j == ind2):
                                    continue
                                for j2 in range(subchunk_qtys2[j]):
                                    dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1],subchunk_embnums2[subchunk_starts2[j] + j2]]
                                    dotm[INTER_NO] += 1
                                    dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2]],subchunk_embnums2[subchunk_starts2[j] + j2]]
                                    dotm[INTER_NO] += 1
                            dotm[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind2]],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1]]
                            dotm[INTER_NO] += 1
                            subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1] = sent_qty2 + i2 - 4
                            subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1]] = sent_qty + i2 - 1
                            subchunk_qtys2[ind1] += 1
                            subchunk_qtys2[ind2] -= 1
                            subchunk_starts2[ind2] += 1
                            dotp = np.zeros(4)
                            dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1]-2]]
                            dotp[INTRA_NO] += 1
                            for j1 in range(subchunk_qtys2[ind1] - 2):
                                dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + j1],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1]-1]]
                                dotp[INTRA_NO] += 1
                                dotp[INTRA] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + j1],subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1]-2]]
                                dotp[INTRA_NO] += 1
                            for j in range(chunk_qty):
                                if j == ind1:
                                    continue
                                for j2 in range(subchunk_qtys2[j]):
                                    dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 1],subchunk_embnums2[subchunk_starts2[j] + j2]]
                                    dotp[INTER_NO] += 1
                                    dotp[INTER] += sent_dots[subchunk_embnums2[subchunk_starts2[ind1] + subchunk_qtys2[ind1] - 2],subchunk_embnums2[subchunk_starts2[j] + j2]]
                                    dotp[INTER_NO] += 1
                            dot2 = dot2 - dotm + dotp
                            m_intra_dot = dot2[INTRA] / dot2[INTRA_NO] if dot2[INTRA_NO] != 0 else 0
                            m_inter_dot = dot2[INTER] / dot2[INTER_NO] if dot2[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys2[ind]
                                subchunk_qtys_r[1] = subchunk_qtys2[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys2[ind2]
                                subchunk_embnums_r[0:subchunk_qtys2[ind]] = subchunk_embnums2[subchunk_starts2[ind]:subchunk_starts2[ind]+subchunk_qtys2[ind]]
                                subchunk_embnums_r[subchunk_qtys2[ind]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]] = subchunk_embnums2[subchunk_starts2[ind1]:subchunk_starts2[ind1]+subchunk_qtys2[ind1]]
                                subchunk_embnums_r[subchunk_qtys2[ind]+subchunk_qtys2[ind1]:subchunk_qtys2[ind]+subchunk_qtys2[ind1]+subchunk_qtys2[ind2]] = subchunk_embnums2[subchunk_starts2[ind2]:subchunk_starts2[ind2]+subchunk_qtys2[ind2]]
                                subchunk_starts_r[0] = subchunk_starts2[ind1]
                                subchunk_starts_r[1] = subchunk_starts2[ind2]
                                dot = np.copy(dot2)
                                update = 1
                        else:
                            dotm = np.zeros(4)
                            for j1 in range(1, subchunk_qtys0[ind2]):
                                dotm[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2] + j1],subchunk_embnums0[subchunk_starts0[ind2]]]
                                dotm[INTRA_NO] += 1
                            for j in range(chunk_qty):
                                if j == ind2:
                                    continue
                                for j2 in range(subchunk_qtys0[j]):
                                    dotm[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind2]],subchunk_embnums0[subchunk_starts0[j] + j2]]
                                    dotm[INTER_NO] += 1
                            subchunk_qtys0[ind1] += 1
                            subchunk_qtys0[ind2] -= 1
                            subchunk_starts0[ind2] += 1
                            dotp = np.zeros(4)
                            for j1 in range(subchunk_qtys0[ind1] - 1):
                                dotp[INTRA] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1] + j1],subchunk_embnums0[subchunk_starts0[ind1] + subchunk_qtys0[ind1] - 1]]
                                dotp[INTRA_NO] += 1
                            for j in range(chunk_qty):
                                if j == ind1:
                                    continue
                                for j2 in range(subchunk_qtys0[j]):
                                    dotp[INTER] += sent_dots[subchunk_embnums0[subchunk_starts0[ind1] + subchunk_qtys0[ind1] - 1],subchunk_embnums0[subchunk_starts0[j] + j2]]
                                    dotp[INTER_NO] += 1
                            dot0 = dot0 - dotm + dotp
                            m_intra_dot = dot0[INTRA] / dot0[INTRA_NO] if dot0[INTRA_NO] != 0 else 0
                            m_inter_dot = dot0[INTER] / dot0[INTER_NO] if dot0[INTER_NO] != 0 else 0
                            reward2 = m_intra_dot - m_inter_dot
                            if (reward2 > reward) and ((chunks[ind] != i1) or (chunks[ind1] != i2)):
                                chunks[ind] = i1
                                chunks[ind1] = i2
                                chunk_lengths[ind] = chunk_length1
                                chunk_lengths[ind1] = chunk_length2
                                chunk_lengths[ind2] = chunk_length3
                                reward = reward2
                                subchunk_qtys_r[0] = subchunk_qtys0[ind]
                                subchunk_qtys_r[1] = subchunk_qtys0[ind1]
                                subchunk_qtys_r[2] = subchunk_qtys0[ind2]
                                subchunk_embnums_r[0:subchunk_qtys0[ind]] = subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]]
                                subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]] = subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]]
                                subchunk_embnums_r[subchunk_qtys0[ind]+subchunk_qtys0[ind1]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]+subchunk_qtys0[ind2]] = subchunk_embnums0[subchunk_starts0[ind2]:subchunk_starts0[ind2]+subchunk_qtys0[ind2]]
                                subchunk_starts_r[0] = subchunk_starts0[ind1]
                                subchunk_starts_r[1] = subchunk_starts0[ind2]
                                dot = np.copy(dot0)
                                update = 1
            
            subchunk_qtys0[ind] = subchunk_qtys_r[0]
            subchunk_qtys0[ind1] = subchunk_qtys_r[1]
            subchunk_qtys0[ind2] = subchunk_qtys_r[2]
            subchunk_starts0[ind1] = subchunk_starts_r[0]
            subchunk_starts0[ind2] = subchunk_starts_r[1]
            subchunk_embnums0[subchunk_starts0[ind]:subchunk_starts0[ind]+subchunk_qtys0[ind]] = subchunk_embnums_r[0:subchunk_qtys0[ind]]
            subchunk_embnums0[subchunk_starts0[ind1]:subchunk_starts0[ind1]+subchunk_qtys0[ind1]] = subchunk_embnums_r[subchunk_qtys0[ind]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]]
            subchunk_embnums0[subchunk_starts0[ind2]:subchunk_starts0[ind2]+subchunk_qtys0[ind2]] = subchunk_embnums_r[subchunk_qtys0[ind]+subchunk_qtys0[ind1]:subchunk_qtys0[ind]+subchunk_qtys0[ind1]+subchunk_qtys0[ind2]]
            subchunk_qtys1[ind] = subchunk_qtys_r[0]
            subchunk_qtys1[ind1] = subchunk_qtys_r[1]
            subchunk_qtys1[ind2] = subchunk_qtys_r[2]
            subchunk_qtys2[ind] = subchunk_qtys_r[0]
            subchunk_qtys2[ind1] = subchunk_qtys_r[1]
            subchunk_qtys2[ind2] = subchunk_qtys_r[2]
            dot0 = np.copy(dot)

        if update == 0:
            break   
    result = {}
    result[0] = chunks
    result[1] = reward
    result[2] = update
    return result

def chunking_init_3(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_embeddings):
    dim_emb = 1536

    chunk_ends = np.zeros(sent_qty, dtype=np.int32)
    chunk_lengths = np.zeros(sent_qty, dtype=np.int32)
    chunk_embeddings =np.zeros((sent_qty, dim_emb))

    sent3_qty = 0
    sent3_length = 0
    sent_cnt = 0
    cnt3 = 0
    while sent_cnt < sent_qty:
        if (sent3_length + sent_lengths[sent_cnt] <= chunk_size) and (cnt3 < 3):
            sent3_length += sent_lengths[sent_cnt]
            chunk_embeddings[sent3_qty] = chunk_embeddings[sent3_qty] + sent_embeddings[sent_cnt]
            sent_cnt += 1
            cnt3 += 1
        else:
            chunk_lengths[sent3_qty] = sent3_length
            chunk_ends[sent3_qty] = sent_cnt - 1
            chunk_embeddings[sent3_qty] = chunk_embeddings[sent3_qty] / cnt3
            sent3_qty += 1
            sent3_length = 0
            cnt3 = 0

    if cnt3 != 0:
        chunk_lengths[sent3_qty] = sent3_length
        chunk_ends[sent3_qty] = sent_cnt - 1
        chunk_embeddings[sent3_qty] = chunk_embeddings[sent3_qty] / cnt3
        sent3_qty += 1

    for i in range(sent3_qty, chunk_qty, -1):
        max_dot = -1
        max_j = 0
        for j in range(i-1):
            if chunk_lengths[j] + chunk_lengths[j+1] <= chunk_size:
                jdot = np.dot(chunk_embeddings[j], chunk_embeddings[j+1])
                if max_dot < jdot:
                    max_dot = jdot
                    max_j = j
        chunk_start = 0 if max_j == 0 else chunk_ends[max_j-1]
        chunk_embeddings[max_j] = (chunk_embeddings[max_j] *(chunk_ends[max_j] - chunk_start) + chunk_embeddings[max_j+1] *(chunk_ends[max_j+1] - chunk_ends[max_j])) / (chunk_ends[max_j + 1] - chunk_start)
        chunk_ends[max_j] = chunk_ends[max_j+1]
        chunk_lengths[max_j] += chunk_lengths[max_j+1]
        for j in range(max_j+1, i-1):
            chunk_ends[j] = chunk_ends[j+1] 
            chunk_lengths[j] = chunk_lengths[j+1]
            chunk_embeddings[j] = chunk_embeddings[j+1]
    
    chunks = chunk_ends[0:chunk_qty]
    return chunks

def chunking_init_4(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_embeddings):
    dim_emb = 1536
    sent_qty2 = sent_qty + sent_qty - 1

    chunk_ends = np.zeros(sent_qty, dtype=np.int32)
    chunk_lengths = np.zeros(sent_qty, dtype=np.int32)
    chunk_embeddings =np.zeros((sent_qty, dim_emb))

    sent3_qty = 0
    sent3_length = 0
    sent_cnt = 0
    cnt3 = 0
    while sent_cnt < sent_qty:
        if (sent3_length + sent_lengths[sent_cnt] <= chunk_size) and (cnt3 < 3):
            sent3_length += sent_lengths[sent_cnt]
            sent_cnt += 1
            cnt3 += 1
        else:
            chunk_lengths[sent3_qty] = sent3_length
            chunk_ends[sent3_qty] = sent_cnt - 1
            if cnt3 == 1:
                chunk_embeddings[sent3_qty] = sent_embeddings[sent_cnt-1]
            elif cnt3 == 2:
                chunk_embeddings[sent3_qty] = sent_embeddings[sent_qty + sent_cnt - 2]
            elif cnt3 == 3:
                chunk_embeddings[sent3_qty] = sent_embeddings[sent_qty2 + sent_cnt - 3]
            sent3_qty += 1
            sent3_length = 0
            cnt3 = 0
    if cnt3 != 0:
        chunk_lengths[sent3_qty] = sent3_length
        chunk_ends[sent3_qty] = sent_cnt - 1
        if cnt3 == 1:
            chunk_embeddings[sent3_qty] = sent_embeddings[sent_cnt-1]
        elif cnt3 == 2:
            chunk_embeddings[sent3_qty] = sent_embeddings[sent_qty + sent_cnt - 2]
        elif cnt3 == 3:
            chunk_embeddings[sent3_qty] = sent_embeddings[sent_qty2 + sent_cnt - 3]
        sent3_qty += 1
        
    for i in range(sent3_qty, chunk_qty, -1):
        max_dot = -1
        max_j = 0
        for j in range(i-1):
            if chunk_lengths[j] + chunk_lengths[j+1] <= chunk_size:
                jdot = np.dot(chunk_embeddings[j], chunk_embeddings[j+1])
                if max_dot < jdot:
                    max_dot = jdot
                    max_j = j
        chunk_start = 0 if max_j == 0 else chunk_ends[max_j-1]
        chunk_embeddings[max_j] = (chunk_embeddings[max_j] *(chunk_ends[max_j] - chunk_start) + chunk_embeddings[max_j+1] *(chunk_ends[max_j+1] - chunk_ends[max_j])) / (chunk_ends[max_j + 1] - chunk_start)
        chunk_ends[max_j] = chunk_ends[max_j+1]
        chunk_lengths[max_j] += chunk_lengths[max_j+1]
        for j in range(max_j+1, i-1):
            chunk_ends[j] = chunk_ends[j+1]
            chunk_lengths[j] = chunk_lengths[j+1]
            chunk_embeddings[j] = chunk_embeddings[j+1]
    
    chunks = chunk_ends[0:chunk_qty]
    return chunks

async def _chunking(text, chunk_size, chunk_qty):

    sentences = nltk.sent_tokenize(text)
    sent_qty = len(sentences)
    sent_lengths = np.zeros(sent_qty, dtype=np.int32)
    for i in range(sent_qty):
        sent_lengths[i] = len(sentences[i]) + 1
    for i in range(sent_qty - 1):
        sent1 = " ".join(sentences[i:i+2])
        sentences.append(sent1)
    for i in range(sent_qty - 2):
        sent1= " ".join(sentences[i:i+3])
        sentences.append(sent1)
    
    client = AsyncOpenAI(api_key="sk-")
    res = await client.embeddings.create(
        input=sentences, 
        model="text-embedding-ada-002")
    sent_embeddings = np.array([item.embedding for item in res.data])
    sent_qty3 = sent_qty * 3 - 3
    # sent_dots = np.dot(np.asarray(sent_embeddings[:sent_qty3]), np.asarray(sent_embeddings[:sent_qty3]).T)
    sent_dots = sent_embeddings @ sent_embeddings.T

    
    chunks_3 = chunking_init_3(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_embeddings)
    updres_3 = chunking_update_1(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks_3, 3, 1)
    chunks_4 = chunking_init_4(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_embeddings)
    updres_4 = chunking_update_1(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks_4, 3, 1)
    chunks0 = chunks_3 if updres_3[1] > updres_4[1] else chunks_4
    time_limit = 6
    start_time = time.time()
    chunks1 = chunks0
    for upd in range(10):
        time1 = time.time()
        result = chunking_update_2(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks1, 3, 2)
        chunks1 = result[0]
        time2 = time.time()
        if (result[2] == 0) or (time2 + (time2- time1) - start_time > time_limit):
            break
    reward1 = result[1]
    chunks = chunks1
    if time2 -start_time < time_limit * 0.5:
        chunks2 = chunks0
        for upd in range(10):
            time1 = time.time()
            result = chunking_update_2(chunk_size, chunk_qty, sent_lengths, sent_qty, sent_dots, chunks2, 3, 3)
            chunks2 = result[0]
            time2 = time.time()
            if (result[2] == 0) or (time2 + (time2- time1) - start_time > time_limit):
                break
        reward2 = result[1]
        if reward2 > reward1:
            chunks = chunks2

    chunk_texts = []
    for i in range(chunk_qty):
        chunk_start = 0 if i == 0 else chunks[i-1] + 1
        sent1 = " ".join(sentences[chunk_start:chunks[i]+1])
        chunk_texts.append(sent1)

    return chunk_texts

async def main():
    index = 1
    # files = [{"name":"1.txt","chunk_size":2000,"num_chunks":32},{"name":"2.txt","chunk_size":4000,"num_chunks":18},{"name":"3.txt","chunk_size":4000,"num_chunks":16},{"name":"4.txt","chunk_size":2000,"num_chunks":32},{"name":"5.txt","chunk_size":3000,"num_chunks":14},{"name":"6.txt","chunk_size":3000,"num_chunks":16}]
    files = [{"name":"1.txt","chunk_size":2000,"num_chunks":30}]

    for i in range(len(files)):

        start_time = time.time()
        with open(files[i]['name'],"r", encoding="utf-8") as file:
            text = file.read()

        chunks = await _chunking(text, chunk_size=files[i]['chunk_size'], chunk_qty=files[i]['num_chunks'])
        chunks_file = []
        for j, chunk in enumerate(chunks):
            print(f"chunk - {j+1} - {chunk}")
            print(len(chunk))
            chunks_file.append({"Chunk Index":j+1,"Chunk Content":chunk})


        df = pd.DataFrame(chunks_file)

        output_file = f"re-{index}.csv"
        df.to_csv(output_file, index=False)  

        print("CSV file saved successfully!")
        end_time = time.time()
        index += 1
        print(f"{files[i]['name']} : {output_file}")
        print(f"{files[i]['name']} took {end_time - start_time}s to complete chunking")

if __name__ == '__main__':
    asyncio.run(main())       
