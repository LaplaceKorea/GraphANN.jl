import os

from benchmark.algorithms.base import BaseANN
from benchmark.datasets import DATASETS, download_accelerated

# Bootstrap importing GraphANN code
import julia
from julia import Pkg
Pkg.activate(os.getenv("PYANN_ROOT"))
from julia import PyANN

class GraphANN(BaseANN):
    def __init__(self, metric, index_params):
        self.name = "GraphANN"
        self._index_params = index_params

        # Manually instantiate the metric
        if metric == "euclidean":
            self._metric = PyANN.Euclidean()
        elif metric == "angular":
            self._metric = PyANN.InnerProduct()
        else:
            print("Error: unknown metric {}".format(metric))

        # Other helpful parameters
        self.PQ = 0 if index_params.get("PQ") == None else index_params.get("PQ")

    #####
    ##### BaseANN API
    #####

    def track(self):
        return "T1"

    def fit(self, dataset):
        raise NotImplementedError()

    def load_index(self, dataset):
        ds = DATASETS[dataset]()
        # Metric
        if ds.distance() == "euclidean":
            self._metric = PyANN.Euclidean()
        elif ds.distance() == "ip":
            self._metric = PyANN.InnerProduct()
        else:
            print("Unsuported distance function.")
            return False

        # Data Type
        if ds.dtype == "float32":
            self._dtype = PyANN.Float32
        elif ds.dtype == "int8":
            self._dtype = PyANN.Int8
        elif ds.dtype == "uint8":
            self._dtype = PyANN.UInt8
        else:
            print ("Unsupported data type.")
            return False

        # Dimensionality
        self._dims = ds.d
        self._index = PyANN.loadindex(
            self.create_index_dir(ds),
            self._dtype,
            self._dims,
            self._metric,
        )
        return True

    def index_files_to_store(self, dataset):
        raise NotImplementedError()

    def query(self, X, k):
        raise NotImplementedError()

    def range_query(self, X, radius):
        raise NotImplementedError()

    def get_results(self):
        return self.res

    def get_range_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    #####
    ##### Utilities
    #####

    def index_name(self):
        return "generic_index"

    def create_index_dir(self, dataset):
        index_dir = os.path.join(os.getcwd(), "data", "indices")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, "T2")
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.__str__())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, dataset.short_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        index_dir = os.path.join(index_dir, self.index_name())
        os.makedirs(index_dir, mode=0o777, exist_ok=True)
        return index_dir


