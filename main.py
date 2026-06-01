from src.loan_fraud_package.pipeline import run_loan_pipeline


def main():
    """
    Entry point for running Loan ML pipeline
    """

    print("Starting Loan ML Pipeline...")

    model = run_loan_pipeline()

    print("Pipeline Execution Completed Successfully")


if __name__ == "__main__":
    main()