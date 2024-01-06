import pandas as pd
import models.decisions_tree as tree
import models.gaussian_naive_bayes as gnb
import models.k_nearest_neighbors as knn
import models.nearest_neighbor as neigh
import models.random_forest as rf
import models.stochastic_gradient_descent as sgd
import models.svm_gaussian as svm_gsn
import models.svm_linear as svm_ln
import models.svm_polynomial as svm_ply
import models.svm_sigmoid as svm_sgm
if __name__ == '__main__':
    models_accuracies = [
    tree.create_decisions_tree_model("hold_out", "hu"),
    tree.create_decisions_tree_model("10_cross", "hu"),
    tree.create_decisions_tree_model("hold_out", "zernike"),
    tree.create_decisions_tree_model("10_cross", "zernike"),
    gnb.create_gaussian_naive_bayes_model("hold_out", "hu"),
    gnb.create_gaussian_naive_bayes_model("10_cross", "hu"),
    gnb.create_gaussian_naive_bayes_model("hold_out", "zernike"),
    gnb.create_gaussian_naive_bayes_model("10_cross", "zernike"),
    knn.create_nearest_neighbors_model("hold_out", "hu"),
    knn.create_nearest_neighbors_model("10_cross", "hu"),
    knn.create_nearest_neighbors_model("hold_out", "zernike"),
    knn.create_nearest_neighbors_model("10_cross", "zernike"),
    neigh.create_nearest_neighbor_model("hold_out", "hu"),
    neigh.create_nearest_neighbor_model("10_cross", "hu"),
    neigh.create_nearest_neighbor_model("hold_out", "zernike"),
    neigh.create_nearest_neighbor_model("10_cross", "zernike"),
    rf.create_random_forest_model("hold_out", "hu"),
    rf.create_random_forest_model("10_cross", "hu"),
    rf.create_random_forest_model("hold_out", "zernike"),
    rf.create_random_forest_model("10_cross", "zernike"),
    sgd.create_stochastic_gradient_descent_model("hold_out", "hu"),
    sgd.create_stochastic_gradient_descent_model("10_cross", "hu"),
    sgd.create_stochastic_gradient_descent_model("hold_out", "zernike"),
    sgd.create_stochastic_gradient_descent_model("10_cross", "zernike"),
    svm_gsn.create_svm_gaussian_model("hold_out", "hu"),
    svm_gsn.create_svm_gaussian_model("10_cross", "hu"),
    svm_gsn.create_svm_gaussian_model("hold_out", "zernike"),
    svm_gsn.create_svm_gaussian_model("10_cross", "zernike"),
    svm_ln.create_svm_linear_model("hold_out", "hu"),
    svm_ln.create_svm_linear_model("10_cross", "hu"),
    svm_ln.create_svm_linear_model("hold_out", "zernike"),
    svm_ln.create_svm_linear_model("10_cross", "zernike"),
    svm_ply.create_svm_polynomial_model("hold_out", "hu"),
    svm_ply.create_svm_polynomial_model("10_cross", "hu"),
    svm_ply.create_svm_polynomial_model("hold_out", "zernike"),
    svm_ply.create_svm_polynomial_model("10_cross", "zernike"),
    svm_sgm.create_svm_sigmoid_model("hold_out", "hu"),
    svm_sgm.create_svm_sigmoid_model("10_cross", "hu"),
    svm_sgm.create_svm_sigmoid_model("hold_out", "zernike"),
    svm_sgm.create_svm_sigmoid_model("10_cross", "zernike")]
    accuracies = pd.DataFrame(models_accuracies)
    print("Accuracies: ")
    print(accuracies.sort_values("Accuracy").tail())