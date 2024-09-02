import os
from airflow import DAG
from airflow.decorators import dag, task_group, task
from airflow.datasets import Dataset
from airflow.operators.empty import EmptyOperator
from airflow.models.baseoperator import chain
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
from datetime import datetime
import pandas as pd
import io
import re

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

@dag(
    dag_id='Access_and_transform_data',
    default_args=default_args,
    description='Extract data from S3, transform it, join it, load to Snowflake, and upload to S3',
    schedule=None,
    catchup=False,
)
def access_data_and_transform():

    @task_group(group_id="Read_Data")
    def read_data_from_s3():
        @task
        def read_discord_data():
            s3_hook = S3Hook(aws_conn_id='aws_conn')
            data = s3_hook.read_key(key='viggo-data/train/subset-part1.jsonl', bucket_name='astronomer-anyscale-demo-2')
            df = pd.read_json(io.StringIO(data), lines=True)
            return {'source': 'discord', 'data': df.to_dict()}


        @task
        def read_twitch_data():
            s3_hook = S3Hook(aws_conn_id='aws_conn')
            data = s3_hook.read_key(key='viggo-data/train/subset-part2.jsonl', bucket_name='astronomer-anyscale-demo-2')
            df = pd.read_json(io.StringIO(data), lines=True)
            return {'source': 'twitch', 'data': df.to_dict()}

        discord_data = read_discord_data()
        twitch_data = read_twitch_data()
        return [discord_data, twitch_data]

    @task
    def write_metrics_to_snowflake(data_list: list):
        metrics = []
        for data_dict in data_list: 
            source = data_dict['source']
            df = pd.DataFrame(data_dict['data'])
            record_count = len(df)
            metrics.append({'Source': source, 'RecordCount': record_count})
        
        metrics_df = pd.DataFrame(metrics)
        
        snowflake_hook = SnowflakeHook(snowflake_conn_id='snowflake_conn')
        # Update the metrics for each source
        for _, row in metrics_df.iterrows():
            update_query = f"""
            MERGE INTO ASTRONOMER_ANYSCALE_DEMO target
            USING (SELECT '{row['Source']}' AS Source, {row['RecordCount']} AS NewCount) source
            ON target.Source = source.Source
            WHEN MATCHED THEN
                UPDATE SET RecordCount = target.RecordCount + source.NewCount
            WHEN NOT MATCHED THEN
                INSERT (Source, RecordCount) VALUES (source.Source, source.NewCount);
            """
            snowflake_hook.run(update_query)
        
        print(f"Updated source metrics: {metrics}")

    @task
    def transform_data(data_dict: dict):
        source = data_dict['source']
        data = data_dict['data']
        # Convert dictionary back to DataFrame
        df = pd.DataFrame(data)
        
        return {
            'transformed_data': df,
            'source': source
        }

    @task
    def join_data(transformed_data_list: list):
        discord_df = next(item['transformed_data'] for item in transformed_data_list if item['source'] == 'discord')
        twitch_df = next(item['transformed_data'] for item in transformed_data_list if item['source'] == 'twitch')
        
        # Concatente the data
        joined_df = pd.concat([discord_df, twitch_df], axis=0)
        
        # Save the joined DataFrame to a buffer
        buffer = io.StringIO()
        joined_df.to_csv(buffer, index=False)
        buffer.seek(0)
        
        return {
            'joined_data': buffer.getvalue(),
            'source': 'joined'
        }

    @task
    def upload_to_s3(data_dict: dict):
        # Convert the joined data back to a DataFrame
        joined_df = pd.read_csv(io.StringIO(data_dict['joined_data']))
        
        # Convert DataFrame to JSONL
        jsonl_buffer = io.StringIO()
        joined_df.to_json(jsonl_buffer, orient='records', lines=True)
        jsonl_buffer.seek(0)
        
        # Upload to S3
        s3_hook = S3Hook(aws_conn_id='aws_conn')
        s3_hook.load_string(
            string_data=jsonl_buffer.getvalue(),
            key=f'viggo-data/transformed/{data_dict["source"]}_data.jsonl',
            bucket_name='astronomer-anyscale-demo-2',
            replace=True
        )

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", outlets=[Dataset("data_transformed")])

    data_read_tg = read_data_from_s3()
    write_metrics_to_snowflake_task = write_metrics_to_snowflake(data_list=data_read_tg)
    transformed_data = transform_data.expand(data_dict=data_read_tg)
    joined_data = join_data(transformed_data_list=transformed_data)
    upload_task = upload_to_s3(data_dict=joined_data)

    chain(start, data_read_tg, write_metrics_to_snowflake_task, end)
    chain(data_read_tg, transformed_data, joined_data, upload_task, end)

dag = access_data_and_transform()