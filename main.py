import logging
import csv
import subprocess

from config.constants import model_dict

from svc.bert.handler import BertFillMaskService
from svc.roberta.handler import RobertaFillMaskService
from svc.t5.handler import T5TranslationService

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


fieldnames = ["Model_Name", "Tokenizer_Load_Time", "Model_Load_Time", "Inference_Time"]
rows = []


def run_bert_benchmark(model_name):
    input = "The largest city in United States of America by population is [MASK]."
    service = BertFillMaskService()
    prediction, metrics = service.fill_mask(model_name=model_name, text=input)

    row = {
        "Model_Name": model_name,
        "Tokenizer_Load_Time": metrics["TokenizerLoadTime"],
        "Model_Load_Time": metrics["ModelLoadTime"],
        "Inference_Time": metrics["PredictionTime"],
    }

    rows.append(row)
    logger.info("Results and Metrics for BERT : " + model_name)
    logger.info("Prompt : " + input)
    logger.info("Output : " + prediction)


def run_roberta_benchmark(model_name):
    input = "The capital of France is <mask>."
    service = RobertaFillMaskService()
    prediction, metrics = service.fill_mask(model_name=model_name, text=input)
    row = {
        "Model_Name": model_name,
        "Tokenizer_Load_Time": metrics["TokenizerLoadTime"],
        "Model_Load_Time": metrics["ModelLoadTime"],
        "Inference_Time": metrics["PredictionTime"],
    }

    rows.append(row)
    logger.info("Results and Metrics for BERT : " + model_name)
    logger.info("Prompt : " + input)
    logger.info("Output : " + prediction)


def run_t5_benchmark(model_name):
    input = "translate English to German: Tell me about Berlin."
    service = T5TranslationService()
    translation, metrics = service.fill_mask(model_name=model_name, text=input)
    row = {
        "Model_Name": model_name,
        "Tokenizer_Load_Time": metrics["TokenizerLoadTime"],
        "Model_Load_Time": metrics["ModelLoadTime"],
        "Inference_Time": metrics["PredictionTime"],
    }

    rows.append(row)
    logger.info("Results and Metrics for BERT : " + model_name)
    logger.info("Prompt : " + input)
    logger.info("Output : " + translation)


model_runner = {
    "bert": run_bert_benchmark,
    "roberta": run_roberta_benchmark,
    "t5": run_t5_benchmark,
}


def calc_average(sum, count):
    return round((sum / count), 5)


def aggregate_values(arr):
    aggregate = {}
    for _, item in enumerate(arr):
        print(item["Model_Name"])
        if item["Model_Name"] not in aggregate:
            aggregate[item["Model_Name"]] = {
                "Tokenizer_Load_Time": {"sum": 0, "count": 0},
                "Model_Load_Time": {"sum": 0, "count": 0},
                "Inference_Time": {"sum": 0, "count": 0},
            }

        aggregate[item["Model_Name"]] = {
            "Tokenizer_Load_Time": {
                "sum": aggregate[item["Model_Name"]]["Tokenizer_Load_Time"]["sum"]
                + item["Tokenizer_Load_Time"],
                "count": aggregate[item["Model_Name"]]["Tokenizer_Load_Time"]["count"]
                + 1,
            },
            "Model_Load_Time": {
                "sum": aggregate[item["Model_Name"]]["Model_Load_Time"]["sum"]
                + item["Model_Load_Time"],
                "count": aggregate[item["Model_Name"]]["Model_Load_Time"]["count"] + 1,
            },
            "Inference_Time": {
                "sum": aggregate[item["Model_Name"]]["Inference_Time"]["sum"]
                + item["Inference_Time"],
                "count": aggregate[item["Model_Name"]]["Inference_Time"]["count"] + 1,
            },
        }

    agg_list = []
    for model_name, _metrics in aggregate.items():
        agg_list.append(
            {
                "Model_Name": model_name,
                "Tokenizer_Load_Time": calc_average(
                    _metrics["Tokenizer_Load_Time"]["sum"],
                    _metrics["Tokenizer_Load_Time"]["count"],
                ),
                "Model_Load_Time": calc_average(
                    _metrics["Model_Load_Time"]["sum"],
                    _metrics["Model_Load_Time"]["count"],
                ),
                "Inference_Time": calc_average(
                    _metrics["Inference_Time"]["sum"],
                    _metrics["Inference_Time"]["count"],
                ),
            }
        )

    return agg_list


def run_benchmark():
    num_iterations = 1
    for _ in range(num_iterations):
        for category, models in model_dict.items():
            for _, model in enumerate(models):
                # logger.info("Clearing Cache")
                # subprocess.run(["./clear_cache.sh"], shell=True)
                model_runner[category](model)

    agg_list = aggregate_values(rows)
    print(agg_list)

    with open("results.csv", "w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=fieldnames)

        writer.writeheader()
        for _, _row in enumerate(agg_list):
            writer.writerow(_row)


if __name__ == "__main__":
    run_benchmark()
