from azure.ai.ml.entities import BatchJob, Input

endpoint_name = "doc-classifier-batch"
deployment_name = "deploy-en-model"  # nazwa z YAML (type:model)

job = BatchJob(
    name="run-en-001",
    endpoint_name=endpoint_name,
    deployment_name=deployment_name,
    inputs={
        "input_data": Input(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/in/en/"
        )
    },
    outputs={
        "output_data": Input(
            type="uri_folder",
            path="azureml://datastores/workspaceblobstore/paths/out/en/"
        )
    }
)

created = ml_client.batch_jobs.begin_create_or_update(job).result()
print("Submitted:", created.name)
