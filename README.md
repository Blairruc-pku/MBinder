# MBinder
Repo of the VLDB 2026 submission paper: Binding Pre-trained Models to Your Data! Efficient Model Selection and Adaptation  for Online In-DBMS Machine Learning. 
## Introduction
MBinder is a solution to leverage pre-trained models within the DBMS in the UDF manner. It achieves high speed and accuracy compared to other in-DBMS ML methods through two major components: Model selection and model synchronization. Model selection conducts a two-phased method to select an appropriate model for the given task, while the model synchronization method updates the model with increamental data to maintain model freshness. \
![image](./MBinder/figure/Intro.png)
## Performance
The following table demonstrates the overall performance of MBinders. MBinder saves large amount of time while achieving excellent performance on accuracy. \
![image](./MBinder/figure/overall.png)

## Quick Start
MBinder is built on PostgreSQL 14.9 and Python 3.9. Note that users need to activate plpython3u extension in PostgreSQL to support ML UDFs of MBinder. After PostgreSQL and Python setups, load the UDFs and models into the PostgreSQL and MBinder is ready to bind appropriate model to the given task. MBinder supports pre-trained models that `git clone` from HuggingFace, since its model execution logic is built on *transformers* package.\
**1. Model Selection** \
Suppose we have already created a table named *agnews* with contents like: 
| text | label |
| ---- | ---- |
| Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again. | 0 |
| Video games 'good for children' Computer games can promote problem-solving and team-building in children, say games industry experts. | 1 |

MBinder is able to quickly select a pre-trained model for the table by:
``` SQL
CALL ModelSelection('agnews');
```
This UDF then starts to select models with codes in /MBinder/src/selection. \
**2. Model Synchronization** \
Suppose a model *bert_agnews* has been bound to the table *agnews*, and this table is slightl changed by DML:
``` SQL
INSERT INTO agnews VALUES('Kobe Bryant committed a foot foul from the free throw line with 4.4 seconds left as the Washington Wizards snapped an 11 game road losing streak by beating Los Angeles Lakers 120-116.', 2)
```
One can manually activate model synchronization function to update the model to the latest table:
```SQL
CALL update_model_mix('bert_agnews', 'agnews');
```
**3. Other Methods** \
We also provide UDFs other methods in this repo. For example, `/MBinder/src/selection/UDFs/TRAILS.sql` is a model selection UDF that select models with TRAILS's method. \
