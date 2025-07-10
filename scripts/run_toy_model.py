from FEniCS_simulations.toy_model.pipeline import GPToyModelPipeline

def main():
    pipeline = GPToyModelPipeline(train_gp=False)
    pipeline.run()

if __name__ == "__main__":
    main()
