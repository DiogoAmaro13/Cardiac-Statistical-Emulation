import argparse
from pathlib import Path
from gpr_modelling.forward.pipeline import GPRForwardPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run the GPR forward prediction pipeline.")


    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to input data file (default: datasets.xlsx in PROCESSED_DATA_DIR)"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory to save predictions and plots (default: FORWARD_RESULTS_DIR)"
    )

    return parser.parse_args()


def main():
    #args = parse_args()

    pipeline = GPRForwardPipeline(
        #data_path=Path(args.data_path) if args.data_path else None,
        #results_dir=Path(args.results_dir) if args.results_dir else None
    )

    pipeline.run()

if __name__ == "__main__":
    main()



