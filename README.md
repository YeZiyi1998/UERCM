# UERCM
These are the datasets and implementation for the paper:

*Ziyi Ye, Xiaohui Xie, Yiqun Liu, Zhihong Wang, Xuesong Chen, Min Zhang, and Shaoping Ma, 2022. In WWW'22.*

Please cite the paper if you use these datasets or codes.

```
@inproceedings{ye2022towards
  title={Towards a Better Understanding of Human Reading Comprehension with Brain Signals},
  author={Ye, Ziyi and Xie, Xiaohui and Liu, Yiqun and Wang, Zhihong and Chen, Xuesong and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={},
  year={2022},
  organization={ACM}
}
```
If you have any problem about this work or dataset, please contact with Ziyi Ye at Yeziyi1998@gmail.com or Yiqun Liu at yiqunliu@tsinghua.edu.cn.

## dataset
In the user study, participants are recruited to perform several reading comprehension tasks. 
Each trial includes a factoid question and the following sentence with graded relevance. 
Under a controlled user study setup in the prevention of potentially confusing effects, EEG data is recorded during the reading process. 
The dataset can be downloaded in https://cloud.tsinghua.edu.cn/d/4ede7ce124cc46f3b42e/.

**raw_txt**:

{i}.txt (i=0,1,2) is the raw experimental reading comprehension tasks, the jth people uses the {j%3}th experimental reading comprehension tasks.
In each line, a dict is provided, where 'q' denotes the reading comprehension text, 'd' is the splited text, 'answer' is the answer, and 'dtype' denotes the generation strategy of the text. 
If 'info' is a key of the dict, it means the task leads to a special test (detailed in our paper), and the value is the content and right choice of the special test.

{i}_process_xl.txt (i=0,1,2) is the annotation results with three external assessors.
The annotations include sentencelevel graded relevance, answer words, and semantic-related words.
The No.{k} line in {i}_process_xl.txt is associated with the No.{k+5} line in {i}.txt (as detailed in the paper, the first five tasks is the training procedure, and thus is not included in our study).
In each line, the word denotes the sentencelevel graded relevance, and the list is the annotation result of answer words, and semantic-related words.
'truth', 'bm25', and 'random' denotes perfectly relevant, relevant, and irrelevant, respectively.
'1', '2', '3', and '5' denotes ordinary word, semantic-related word, answer word, and stop word, respectively. 

**raw_eeg**:
qa{i}.fif (i=0,1,2,...,20) is the processed eeg data of the ith participant. 
The processed procedures are elobrated in our paper.
To achieve higher reproducibility, we provide example code to paint ERP in the **example** directory.

**processed_eeg**:

## example
We provide example code to paint ERP and the feature selection procedures for the classification task.

**paint_erp.py**:
Example code to paint ERP.

**feature_selection.py**:
Example code to preprocess the data for the classification task.

## UERCM
The implementation of the UERCM model.
In the paper, we report the predictoin results with the best performance, the running scripts are given below.

```
# Example

## Train (answer sentence classification)
python main.py -target s -strategy LOPO -lr 5e-3 -batch_size 8 -erp_type erp_lowered_False -save_dir LOPO_lr0.005_ba8

## Evaluate (answer sentence classification)
python ranker_evaluate.py 

## Train and evaluate (answer extraction) 
python main.py -target w -strategy LOPO -lr 0.01 -batch_size 8 -erp_type erp_lowered_True -dmodel 8

```

