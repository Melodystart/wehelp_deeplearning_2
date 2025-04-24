from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='daily_model_update_pipeline',
    start_date=datetime(2024, 1, 1),
    # schedule='30 12 * * *',
    schedule='*/5 * * * *',
    catchup=False,
) as dag:

  tokenize = BashOperator(
      task_id='run_tokenize',
      bash_command='source /home/ubuntu/torch-env/bin/activate && python /home/ubuntu/wehelp_deeplearning_2/train/tokenizer.py',
  )

  embedding = BashOperator(
      task_id='run_embedding',
      bash_command='source /home/ubuntu/torch-env/bin/activate && python /home/ubuntu/wehelp_deeplearning_2/train/embedding.py',
  )

  multi_class = BashOperator(
      task_id='run_multi_class',
      bash_command='source /home/ubuntu/torch-env/bin/activate && python /home/ubuntu/wehelp_deeplearning_2/train/multi-class.py',
  )

  tokenize >> embedding >> multi_class