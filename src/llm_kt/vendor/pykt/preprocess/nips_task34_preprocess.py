import os
import ast
from functools import lru_cache
import pandas as pd
from .utils import sta_infos, write_txt, change2timestamp,format_list2str



def load_nips_data(primary_data_path,meta_data_dir,task_name):
    """The data downloaded from https://competitions.codalab.org/competitions/25449 
    The document can be downloaded from https://arxiv.org/abs/2007.12061.

    Args:
        primary_data_path (_type_): premary data path
        meta_data_dir (_type_): metadata dir
        task_name (_type_): task_1_2 or task_3_4

    Returns:
        dataframe: the merge df
    """
    print("Start load data")
    answer_metadata_path = os.path.join(meta_data_dir,f"answer_metadata_{task_name}.csv")
    question_metadata_path = os.path.join(meta_data_dir,f"question_metadata_{task_name}.csv")
    student_metadata_path = os.path.join(meta_data_dir,f"student_metadata_{task_name}.csv")
    subject_metadata_path = os.path.join(meta_data_dir,f"subject_metadata.csv")
    
    df_primary = pd.read_csv(primary_data_path)
    print(f"len df_primary is {len(df_primary)}")
    #add timestamp
    df_answer = pd.read_csv(answer_metadata_path)
    # Preserve the upstream converter exactly; repeated date strings share results.
    convert_timestamp = lru_cache(maxsize=None)(change2timestamp)
    df_answer['answer_timestamp'] = df_answer['DateAnswered'].apply(convert_timestamp)
    df_question = pd.read_csv(question_metadata_path)
    # df_student = pd.read_csv(student_metadata_path)
    df_subject = pd.read_csv(subject_metadata_path)
    
    #only keep level 3
    keep_subject_ids = set(df_subject[df_subject['Level']==3]['SubjectId'])
    df_question['SubjectId_level3'] = df_question['SubjectId'].apply(lambda x:set(ast.literal_eval(x))&keep_subject_ids)
    
    
    #merge data
    df_merge = df_primary.merge(df_answer[['AnswerId','answer_timestamp']],how='left')#merge answer time
    df_merge = df_merge.merge(df_question[["QuestionId","SubjectId_level3"]],how='left')#merge question subjects
    df_merge['SubjectId_level3_str'] = df_merge['SubjectId_level3'].apply(lambda x:"_".join([str(i) for i in x]))
    print(f"len df_merge is {len(df_merge)}")
    print("Finish load data")
    print(f"Num of student {df_merge['UserId'].unique().size}")
    print(f"Num of question {df_merge['QuestionId'].unique().size}")
    kcs =[]
    for item in df_merge['SubjectId_level3'].values:
        kcs.extend(item)
    print(f"Num of knowledge {len(set(kcs))}")
    return df_merge

def get_user_inters(df):
    """convert df to user sequences 

    Args:
        df (_type_): the merged df

    Returns:
        List: user_inters
    """
    user_inters = []
    for user, group in df.groupby("UserId", sort=False):
        group = group.sort_values(["answer_timestamp","tmp_index"], ascending=True)

        seq_skills = group['SubjectId_level3_str'].tolist()
        seq_ans = group['IsCorrect'].tolist()
        seq_response_cost = ["NA"]
        seq_start_time = group['answer_timestamp'].tolist()
        seq_problems = group['QuestionId'].tolist()
        seq_len = len(group)
        user_inters.append(
            [[str(user), str(seq_len)],
             format_list2str(seq_problems),
             format_list2str(seq_skills),
             format_list2str(seq_ans),
             format_list2str(seq_start_time),
             format_list2str(seq_response_cost)])
    return user_inters


KEYS = ["UserId", "SubjectId_level3_str", "QuestionId"]

def read_data_from_csv(primary_data_path,meta_data_dir,task_name,write_file):
    stares= []
    df = load_nips_data(primary_data_path,meta_data_dir,task_name)
    
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    df['tmp_index'] = range(len(df))
    df = df.dropna(subset=["UserId","answer_timestamp", "SubjectId_level3_str", "IsCorrect", "answer_timestamp","QuestionId"])
    # ================= 修改开始 =================
    # 2. 【关键修改】强制将 IsCorrect 转为 int，去掉 .0
    df['IsCorrect'] = df['IsCorrect'].astype(int)
    
    # 3. (可选建议) 为了保险，把 QuestionId 也转为 int，防止题目ID也变成 100.0
    df['QuestionId'] = df['QuestionId'].astype(int)
    # ================= 修改结束 =================
    df['answer_timestamp'] = df['answer_timestamp'].astype('int64') 
    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    user_inters = get_user_inters(df)
    write_txt(write_file, user_inters)
    
