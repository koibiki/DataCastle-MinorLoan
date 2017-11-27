from feature_engineering.data_loader import DataLoader
from feature_engineering.null_handler import NullHandler
from feature_engineering.visualize_null import Visualize
from feature_engineering.numeric_discretization import Discretizator

# DataLoader.rank()

# null_handler = NullHandler()
# null_handler.null_count()

# visualize = Visualize()
# visualize.visualize_null()

discretizator = Discretizator()
discretizator.data_discretization()