AZURE ML v2 — ONE NOTEBOOK PIPELINE SUBMISSION

Run `azureml_v2_submit_pipeline.ipynb` to submit the whole pipeline on your EXISTING compute.

Contents:
- azureml_v2_submit_pipeline.ipynb
- conda.yaml (Python 3.11 env for AzureML)
- src/step00_filter_resplit.py
- src/step01_relevance_train.py
- src/step02_label_train_compare.py

Edit the CONFIG cell with:
- subscription, resource group, workspace name
- compute name
- datastore URIs (or local paths if you prefer, but URIs are easiest)
