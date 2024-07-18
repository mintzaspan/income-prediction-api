import argparse
import logging
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="data_upload", project="income-prediction-api")
    artifact = wandb.Artifact(name=args.output_artifact, type=args.output_type)
    artifact.add_file(local_path=args.input_artifact)
    run.log_artifact(artifact)
    artifact.wait()
    logger.info(f"Artifact {args.output_artifact} saved and logged")
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload local data file to Weights & Biases"
    )
    parser.add_argument(
        "input_artifact", type=str, help="Path of input file of CSV type"
    )
    parser.add_argument(
        "output_artifact",
        type=str,
        help="Name to save artifact")
    parser.add_argument("output_type", type=str, help="Type of artifact")
    args = parser.parse_args()

    go(args)
