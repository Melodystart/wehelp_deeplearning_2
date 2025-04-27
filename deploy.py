from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='daily_model_update_pipeline',
    start_date=datetime(2025, 4, 27),
    schedule = '0 0 * * *',
    catchup=False,
) as dag:

  embedding = BashOperator(
      task_id='run_embedding',
      bash_command='source /home/ubuntu/torch-env/bin/activate && python /home/ubuntu/wehelp_deeplearning_2/train/embedding.py',
  )

  multi_class = BashOperator(
      task_id='run_multi_class',
      bash_command='source /home/ubuntu/torch-env/bin/activate && python /home/ubuntu/wehelp_deeplearning_2/train/multi-class.py',
  )

  embedding >> multi_class