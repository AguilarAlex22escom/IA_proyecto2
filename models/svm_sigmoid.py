import pandas as pd, numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from .datasets.datasets import create_datasets

accuracies = []
matrices = []
areas = []

def create_svm_sigmoid_model(split, moment):
    print("Working with Support Vector Machine (Sigmoid kernel)...")
    train_dataset, test_dataset = create_datasets("shapes/train_dataset/", "shapes/test_dataset/", moment)
    svm = SVC(kernel = "sigmoid", C = 1, gamma = "scale")

    x_train = train_dataset.drop(["class_name"], axis = 1)
    x_test = test_dataset.drop(["class_name"], axis = 1)
    y_train = train_dataset["class_name"]
    y_test = test_dataset["class_name"]
    label_encoder = LabelEncoder()

    if split == "hold_out": 
        x_train_hd, _, y_train_hd, _ = train_test_split(
            x_train, y_train, train_size = 0.99, shuffle = True, random_state = 0, stratify = y_train
        )

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_hd)
        x_test_scaled = scaler.fit_transform(x_test)

        model = svm.fit(x_train_scaled, y_train_hd)
        y_pred = model.predict(x_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
       
        y_scores = svm.predict_proba(x_test_scaled)
        y_test_binary = label_binarize(y_test, classes = ['circle', 'star', 'triangle', 'square'])
        area = roc_auc_score(y_test_binary, y_scores)

        for label_index in range(y_pred.size):
            print(f"{label_index + 1}.- Real label: {y_test[label_index]}\n{label_index + 1}.- Predicted label: {y_pred[label_index]}")
        make_printers(accuracy, split, moment, matrix, area)
        return {"Model" : "SVM Sigmoid", "Split method": "Hold out", "Accuracy" : accuracy.round(4), "Moments" : moment}
    
    elif split == "5_cross":
        skf = StratifiedKFold(n_splits = 5)
        scaler = StandardScaler()
        ctr = 0
        for train_index, test_index in skf.split(x_train, y_train):
            ctr += 1
            x_test_skf = x_train.iloc[test_index].drop(x_train.iloc[test_index].index)
            y_test_skf = y_train.iloc[test_index].drop(y_train.iloc[test_index].index)
            x_test_skf = pd.concat([x_test] * (int(np.ceil(200 / len(x_test)))), ignore_index=True).head(200)
            y_test_skf = pd.concat([y_test] * (int(np.ceil(200 / len(y_test)))), ignore_index=True).head(200)
            x_train_scaled = scaler.fit_transform(x_train.iloc[train_index])
            x_test_scaled = scaler.fit_transform(x_test_skf)

            y_train_skf, y_test_skf = y_train.iloc[train_index], y_test_skf

            model = svm.fit(x_train_scaled, y_train_skf)
            y_pred = model.predict(x_test_scaled)
            accuracies.append(accuracy_score(y_test_skf, y_pred))
            matrices.append(confusion_matrix(y_test_skf, y_pred))
            
            y_scores = svm.predict_proba(x_test_scaled)
            y_test_binary = label_binarize(y_test_skf, classes = ['circle', 'star', 'triangle', 'square'])
            areas.append(roc_auc_score(y_test_binary, y_scores))

            if ctr == 4:
                for y_index in range(24):
                    print(f"{y_index + 1}.- Label predicted: {y_pred[y_index]}\n{y_index + 1}.- Real label: {y_test_skf.to_numpy()[y_index]}")
        average_accuracy = np.mean(accuracies)
        average_matrix = np.mean(matrices, axis = 0)
        average_area = np.mean(areas)
        make_printers(average_accuracy, split, moment, average_matrix, average_area)
        return {"Model" : "SVM Sigmoid", "Split method": "5-Fold cross", "Accuracy" : average_accuracy.round(4), "Moments" : moment}
    
    elif split == "10_cross":
        skf = StratifiedKFold(n_splits = 10)
        scaler = StandardScaler()
        ctr = 0
        for train_index, test_index in skf.split(x_train, y_train):
            ctr += 1
            x_test_skf = x_train.iloc[test_index].drop(x_train.iloc[test_index].index)
            y_test_skf = y_train.iloc[test_index].drop(y_train.iloc[test_index].index)
            x_test_skf = pd.concat([x_test] * (int(np.ceil(100 / len(x_test)))), ignore_index=True).head(100)
            y_test_skf = pd.concat([y_test] * (int(np.ceil(100 / len(y_test)))), ignore_index=True).head(100)
            x_train_scaled = scaler.fit_transform(x_train.iloc[train_index])
            x_test_scaled = scaler.fit_transform(x_test_skf)

            y_train_skf, y_test_skf = y_train.iloc[train_index], y_test_skf

            model = svm.fit(x_train_scaled, y_train_skf)
            y_pred = model.predict(x_test_scaled)
            accuracies.append(accuracy_score(y_test_skf, y_pred))
            matrices.append(confusion_matrix(y_test_skf, y_pred))
            
            y_scores = svm.predict_proba(x_test_scaled)
            y_test_binary = label_binarize(y_test_skf, classes = ['circle', 'star', 'triangle', 'square'])
            areas.append(roc_auc_score(y_test_binary, y_scores))

            if ctr == 9:
                for y_index in range(24):
                    print(f"{y_index + 1}.- Label predicted: {y_pred[y_index]}\n{y_index + 1}.- Real label: {y_test_skf.to_numpy()[y_index]}")
        average_accuracy = np.mean(accuracies)
        average_matrix = np.mean(matrices, axis = 0)
        average_area = np.mean(areas)
        make_printers(average_accuracy, split, moment, average_matrix, average_area)
        return {"Model" : "SVM Sigmoid", "Split method": "10-Fold cross", "Accuracy" : average_accuracy.round(4), "Moments" : moment}
    else:
        print("Model selection not correct...\n")
