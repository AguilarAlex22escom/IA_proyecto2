from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from datasets.datasets import *

accuracies = []
matrices = []

def create_decisions_tree_model(split, moment):
    print("Working with Decision tree model...")
    train_dataset, test_dataset = create_datasets("shapes/train_dataset/", "shapes/test_dataset/", moment)
    tree = DecisionTreeClassifier()

    x_train = train_dataset.drop(["class_name"], axis = 1)
    x_test = test_dataset.drop(["class_name"], axis = 1)
    y_train = train_dataset["class_name"]
    y_test = test_dataset["class_name"]

    if split == "hold_out": 
        x_train, _, y_train, _ = train_test_split(
            x_train, y_train, train_size = 0.99, shuffle = True, random_state = 0, stratify = y_train
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)

        model = tree.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        for label_index in range(y_pred.size):
            print(f"{label_index + 1}.- Real label: {y_test[label_index]}\n{label_index + 1}.- Predicted label: {y_pred[label_index]}")
        print(f"Accuracy: {accuracy.round(3)} with {split} and {moment} moments")
        print(f"Confusion matrix with {split} and {moment} moments:\n")
        print(matrix)

    elif split == "10_cross":
        skf = StratifiedKFold(n_splits = 10)
        scaler = StandardScaler()
        for train_index, test_index in skf.split(x_train, y_train):
            x_test_skf = x_train.iloc[test_index].drop(x_train.iloc[test_index].index)
            y_test_skf = y_train.iloc[test_index].drop(y_train.iloc[test_index].index)
            x_test_skf = pd.concat([x_test] * (int(np.ceil(100 / len(x_test)))), ignore_index=True).head(100)
            y_test_skf = pd.concat([y_test] * (int(np.ceil(100 / len(y_test)))), ignore_index=True).head(100)
            x_train_scaled = scaler.fit_transform(x_train.iloc[train_index])
            x_test_scaled = scaler.fit_transform(x_test_skf)

            y_train_skf, y_test_skf = y_train.iloc[train_index], y_test_skf

            model = tree.fit(x_train_scaled, y_train_skf)
            y_pred = model.predict(x_test_scaled)
            accuracies.append(accuracy_score(y_test_skf, y_pred))
            matrices.append(confusion_matrix(y_test_skf, y_pred))
            for y_index in range(len(y_pred)):
                print(f"{y_index + 1}.- Label predicted: {y_pred[y_index]}\n{y_index + 1}.- Real label: {y_test_skf.to_numpy()[y_index]}")
        average_accuracy = np.mean(accuracies)
        average_matrix = np.mean(matrices, axis = 0)
        print(f"Accuracy: {average_accuracy.round(3)} with {split} and {moment} moments")
        print(f"Confusion matrix with {split} and {moment} moments:\n")
        print(average_matrix)
    else:
        print("Model selection not correct...\n")