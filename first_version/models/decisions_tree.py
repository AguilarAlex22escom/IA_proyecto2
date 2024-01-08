from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from moments.hu_moments import *
from moments.zernike_moments import *

accuracies = []
matrices = []

def create_dataset(path, moment):
    data = None
    if moment == "zernike":
        data = create_dataset_zernike(path) 
    elif moment == "hu":
        data = create_dataset_hu(path)
    return data

def create_decisions_tree_model(split, moment):
    dataset = create_dataset("shapes/dataset/", moment)
    tree = DecisionTreeClassifier()

    x = dataset.drop(["class_name"], axis = 1)
    y = dataset["class_name"]

    if split == "hold_out": 
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size = 0.7, shuffle = True, random_state = 0, stratify = y
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)

        model = tree.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        print(f"Accuracy (Linear kernel): {accuracy.round(3)} with {split}")
        print("Confusion matrix (Linear kernel): \n")
        print(matrix)

    elif split == "10_cross":
        skf = StratifiedKFold(n_splits = 10)
        scaler = StandardScaler()
        for train_index, test_index in skf.split(x, y):
            x_train_scaled = scaler.fit_transform(x.iloc[train_index])
            x_test_scaled = scaler.fit_transform(x.iloc[test_index])

            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = tree.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            accuracies.append(accuracy_score(y_test, y_pred))
            matrices.append(confusion_matrix(y_test, y_pred))
        average_accuracy = np.mean(accuracies)
        average_matrix = np.mean(matrices, axis = 0)
        print(f"Accuracy (Linear kernel): {average_accuracy.round(3)} with {split}")
        print("Confusion matrix (Linear kernel): \n")
        print(average_matrix)
    else:
        print("Model selection not correct...\n")

create_decisions_tree_model("hold_out", "hu")
create_decisions_tree_model("10_cross", "hu")
create_decisions_tree_model("hold_out", "zernike")
create_decisions_tree_model("10_cross", "zernike")