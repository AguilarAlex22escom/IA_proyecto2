def make_printers(accuracy, split, moment, matrix, area):
        print(f"Area under curve: {area.round(3)} with {split} and {moment} moments")
        print(f"Accuracy: {accuracy.round(3)} with {split} and {moment} moments")
        print(f"Confusion matrix with {split} and {moment} moments:\n")
        print(matrix)