import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

import pre_processing

# Te importy muszą odpowiadać Twoim klasom (z Twojego projektu)
# Zakładam, że w online działało, więc w batch env też musi je mieć.
from your_pkg.models import LogisticRegression, Transformer  # <- PODMIEŃ jeśli masz inne ścieżki importu


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

relevance_model = None
cd_logreg_model = None
cd_transformer_model = None


def init():
    global relevance_model
    global cd_logreg_model
    global cd_transformer_model

    logger.info("Initializing model...")

    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("AZUREML_MODEL_DIR is not set")

    # debug jak u Ciebie
    try:
        print(f"line 23: {os.listdir(model_dir)}")
        print(f"line 24: {model_dir}")
    except Exception:
        pass

    # 1:1 jak na screenie: wchodzisz do podfolderu "AL"
    model_dir = os.path.join(model_dir, "AL")

    try:
        print(f"line 27: {os.listdir(model_dir)}")
    except Exception:
        pass

    relevance_model = LogisticRegression.load(
        os.path.join(model_dir, "logreg_relevance.joblib")
    )

    cd_logreg_model = LogisticRegression.load(
        os.path.join(model_dir, "logreg_cd.joblib")
    )

    transformer = os.path.join(model_dir, "transformer_model")
    model_cd_transformer_path = os.path.join(transformer, "transformer")
    model_cd_transformer_le_path = os.path.join(transformer, "transformer_le.joblib")

    cd_transformer_model = Transformer.load(
        model_cd_transformer_path,
        model_cd_transformer_le_path
    )

    logger.info("Model initialized successfully.")


# -------------------------
# BATCH ENTRYPOINT
# -------------------------
def run(mini_batch: List[Union[str, Dict[str, Any]]]):
    """
    Batch endpoint:
    - mini_batch to LISTA elementów
    - element może być:
        * ścieżką do pliku (uri_folder/uri_file)
        * dict (np. z mltable)
        * JSON string (rzadziej, ale obsłużymy)
    Zwracamy listę wyników, po jednym na element mini_batch.
    """
    outputs = []

    for item in mini_batch:
        try:
            raw_data = _item_to_raw_json_string(item)
            result = _process_raw_data_1to1(raw_data)
            outputs.append({"predictions": result})
        except Exception as e:
            # w batch lepiej nie wywalać całego joba na jednym elemencie
            logger.exception(f"Failed item: {str(e)}")
            outputs.append({"error": str(e)})

    return outputs


def _item_to_raw_json_string(item: Union[str, Dict[str, Any]]) -> str:
    """
    Normalizuje item do raw JSON string, żeby zachować Twoją logikę 1:1.
    """
    if isinstance(item, dict):
        return json.dumps(item, ensure_ascii=False)

    if isinstance(item, str):
        p = Path(item)
        # jeśli wygląda jak plik i istnieje -> czytaj plik
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8")
        # inaczej traktuj jako JSON string
        return item

    raise TypeError(f"Unsupported item type: {type(item)}")


def _process_raw_data_1to1(raw_data: str) -> Dict[str, Any]:
    """
    To jest Twoje stare run(raw_data) przeniesione 1:1,
    tylko jako funkcja pomocnicza (bo batch run() iteruje po mini_batch).
    """
    try:
        logger.info(f"Received request with data: {raw_data}")

        if not raw_data or raw_data.strip() == "":
            raise ValueError("Bad Request: Request body cannot be empty")

        # Parse the input data
        try:
            request_data = json.loads(raw_data)
        except json.JSONDecodeError:
            raise ValueError("Bad Request: Invalid JSON format!")

        if "document" not in request_data or "num_preds" not in request_data:
            raise ValueError("Bad Request: Invalid input, expected 'document' and 'num_preds'")

        document = request_data["document"]
        num_pred = int(request_data["num_preds"])

        response = inference(document, num_pred)

        logger.info(f"Inference response: {response}")

        # 1:1 jak u Ciebie: zwracasz dict / list / error
        if isinstance(response, dict):
            return response
        elif hasattr(response, "tolist"):
            return response.tolist()
        else:
            raise ValueError("Unexpected response format!")

    except ValueError as ve:
        logger.warning(f"Bad request: {str(ve)}")
        raise ve
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise Exception("Internal server error")


# -------------------------
# TWOJA LOGIKA 1:1
# -------------------------
def is_relevant_customer_demand(
    text: str,
    relevant_proba: float,
    cd_logreg_proba: float,
    cd_transformer_proba: float,
):
    """
    1:1 jak na screenie.
    """
    return (
        len(text.split(" ")) > 2
        and cd_logreg_proba > 0.1
        and (
            (relevant_proba > 0.65 and cd_transformer_proba > 0.9)
            or cd_transformer_proba > 0.95
        )
    )


def inference(document: json, num_cd_predictions: int):
    # load and process document
    start_time = time.time()

    text = pd.Series(
        content["text"] for content in
        document["contentDomain"]["byId"].values()
    )

    text = pre_processing.clean_text(text)
    latency = time.time() - start_time
    print("Time to load and clean document: {}".format(latency))

    # score relevance model
    start_time = time.time()
    all_relevance_predictions = relevance_model.predict_proba(text)
    latency = time.time() - start_time
    print("Time to score relevance model: {}".format(latency))

    # score cd logreg model
    start_time = time.time()
    all_cd_logreg_predictions = (
        cd_logreg_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd logreg model: {}".format(latency))

    # score cd transformer
    start_time = time.time()
    all_cd_transformer_predictions = (
        cd_transformer_model.predict_top_n_labels_with_proba(
            text, num_cd_predictions
        )
    )
    latency = time.time() - start_time
    print("Time to score cd transformer model: {}".format(latency))

    # format results
    start_time = time.time()
    document_demand_predictions = set()

    for (
        content,
        relevance_prediction,
        cd_logreg_predictions,
        cd_transformer_predictions,
    ) in zip(
        document["contentDomain"]["byId"].values(),
        all_relevance_predictions,
        all_cd_logreg_predictions,
        all_cd_transformer_predictions,
    ):
        if is_relevant_customer_demand(
            content["text"],
            relevance_prediction,
            cd_logreg_predictions[0]["proba"],
            cd_transformer_predictions[0]["proba"],
        ):
            document_demand_predictions.add(
                cd_transformer_predictions[0]["label"]
            )

        content.update(
            {
                "relevantProba": relevance_prediction,
                "cdLogregPredictions": cd_logreg_predictions,
                "cdTransformerPredictions": cd_transformer_predictions,
            }
        )

    # update document to include document_demand_predictions
    document["documentDemandPredictions"] = list(document_demand_predictions)

    result = document

    latency = time.time() - start_time
    print("Time to format results: {}".format(latency))

    return result
