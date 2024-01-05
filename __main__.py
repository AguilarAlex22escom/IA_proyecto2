from models.decisions_tree import *
from models.gaussian_naive_bayes import *

if __name__ == '__main__':
    decisions_tree.create_decisions_tree_model("hold_out", "hu")
    decisions_tree.create_decisions_tree_model("10_cross", "hu")
    decisions_tree.create_decisions_tree_model("hold_out", "zernike")
    decisions_tree.create_decisions_tree_model("10_cross", "zernike")
    gaussian_naive_bayes.create_gaussian_naive_bayes_model("hold_out", "hu")
    gaussian_naive_bayes.create_gaussian_naive_bayes_model("hold_out", "zernike")
    gaussian_naive_bayes.create_gaussian_naive_bayes_model("10_cross", "hu")
    gaussian_naive_bayes.create_gaussian_naive_bayes_model("10_cross", "zernike")