// 当前代码：   filter 访问时间：1.5s
// python代码： filter 访问时间：40s
// pybind11代码 filter 访问时间：8s
// 正在尝试

#include <vector>
#include <tuple>
#include <iostream>
#include <limits>
#include <future>
#include <cmath>
#include <algorithm>
#include <functional>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/numpy.hpp>
typedef long long LL;
typedef unsigned long long ULL;
namespace p = boost::python;
namespace np = boost::python::numpy;

// const int cc = 5;

p::tuple process_mul(
    const np::ndarray& distances_idx_np, 
    const np::ndarray& distances_np, 
    const np::ndarray& test_label_left, 
    const np::ndarray& test_label_right,
    const np::ndarray& train_label_np, 
    int query_size, int topk, int cc, int label_dim) 
{
    np::ndarray select_idx = np::empty(p::make_tuple(query_size, topk * cc), np::dtype::get_builtin<int>());
    np::ndarray select_dis = np::empty(p::make_tuple(query_size, topk * cc), np::dtype::get_builtin<float>());

    std::fill((int*)select_idx.get_data(), (int*)select_idx.get_data() + query_size * topk * cc, -1);
    std::fill((float*)select_dis.get_data(), (float*)select_dis.get_data() + query_size * topk * cc, std::numeric_limits<float>::infinity());
    // std::fill((float*)select_dis.get_data(), (float*)select_dis.get_data() + query_size * topk * cc, -1);

    int shp1 = distances_idx_np.shape(1);
    // int size1 = distances_idx_np.shape(0);
    // printf("the shape of np: %d %d\n", size1, shp1);

    int*    distances_idx_data  = reinterpret_cast<int*>(distances_idx_np.get_data());
    float*  distances_data      = reinterpret_cast<float*>(distances_np.get_data());
    int*    test_label_left_data  = reinterpret_cast<int*>(test_label_left.get_data());
    int*    test_label_right_data  = reinterpret_cast<int*>(test_label_right.get_data());
    int*    train_label_data    = reinterpret_cast<int*>(train_label_np.get_data());

    // #pragma omp parallel for
    for (int i = 0; i < query_size; ++i) {
        int idx_offset  = i * shp1;
        int label_offset = i * label_dim;

        int cnt = 0;
        int topkcc = topk * cc;
        int itc = i * topkcc;

        for (int j = 0; j < shp1 && cnt < topkcc; ++j) {
            int id_where = distances_idx_data[idx_offset + j];
            float dis_where = distances_data[idx_offset + j];
            int train_label_offset = id_where * label_dim;
            bool flag = true;

            for(int k=0; k<label_dim; k++)
                if ((train_label_data[train_label_offset+k]<test_label_left_data[label_offset+k]) || (test_label_right_data[label_offset+k] < train_label_data[train_label_offset+k])){
                    flag = false;
                    break;
                }
            
            if (!flag) continue;
            int select_idx_offset = itc + cnt;
            ((int*)select_idx.get_data())[select_idx_offset] = id_where;
            ((float*)select_dis.get_data())[select_idx_offset] = dis_where;
            cnt++;
        }
    }

    return p::make_tuple(select_idx, select_dis);
}

// ------------------------------------------------------- split line -------------------------------------------------------


struct Neighbor {
    int index;
    float distance;
};

float euclidean_distance(const np::ndarray& vec1, const np::ndarray& vec2, int dim) {
    float sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        float diff = ((float*)vec1.get_data())[i] - ((float*)vec2.get_data())[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Function to find the k nearest neighbors
std::vector<int> find_k_nearest_neighbors(
    const np::ndarray& valid_idx, 
    const np::ndarray& candidate_vec,
    const np::ndarray& query_vec, 
    int k) 
{
    std::vector<Neighbor> distances;
    int num_candidates = candidate_vec.shape(0);
    int dim_candidates = candidate_vec.shape(1);
    int* valid_idx_data = reinterpret_cast<int*>(valid_idx.get_data());
    float* candidate_vec_data = reinterpret_cast<float*>(candidate_vec.get_data());

    for (int i = 0; i < num_candidates; ++i) {
        np::ndarray candidate = np::empty(p::make_tuple(dim_candidates), np::dtype::get_builtin<float>());
        int candidate_offset = i * dim_candidates;

        for (int j = 0; j < dim_candidates; ++j) {
            ((float*)candidate.get_data())[j] = candidate_vec_data[candidate_offset + j];
        }
        float dist = euclidean_distance(query_vec, candidate, dim_candidates);
        int index = valid_idx_data[i];
        distances.push_back({index, dist});
    }

    std::nth_element(distances.begin(), distances.begin() + k, distances.end(),
                     [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });

    std::vector<int> indices;
    for (int i = 0; i < k; ++i) {
        indices.push_back(distances[i].index);
    }
    return indices;
}

// The main refinement process function
std::tuple<int, np::ndarray> refine_process(
    int i, 
    const np::ndarray& valid_idx, 
    const np::ndarray& candidate_vec, 
    const np::ndarray& query_vec, 
    int topk) 
{
    std::vector<int> sorted_idx = find_k_nearest_neighbors(valid_idx, candidate_vec, query_vec, topk);

    np::ndarray result = np::empty(p::make_tuple(topk), np::dtype::get_builtin<int>());
    for (int j = 0; j < topk; ++j) {
        ((int*)result.get_data())[j] = sorted_idx[j];
    }
    return std::make_tuple(i, result);
}

// The main refinement function
np::ndarray refine(
    const np::ndarray& idxs, 
    const np::ndarray& train_vec, 
    const np::ndarray& test_vec, 
    int topk) 
{
    int query_size = idxs.shape(0);
    int shp1 = idxs.shape(1);

    np::ndarray select_idx = np::empty(p::make_tuple(query_size, topk), np::dtype::get_builtin<int>());
    std::fill((int*)select_idx.get_data(), (int*)select_idx.get_data() + query_size * topk, -1);

    int*    idxs_data       = reinterpret_cast<int*>(idxs.get_data());
    float*  train_vec_data  = reinterpret_cast<float*>(train_vec.get_data());
    float*  test_vec_data   = reinterpret_cast<float*>(test_vec.get_data());
    int dim = test_vec.shape(1);
    // printf("tag 1\n");

    for (int i = 0; i < query_size; ++i) {
        int idx_offset = i * shp1;
        int query_offset = i * dim;
        np::ndarray query_vec = np::empty(p::make_tuple(dim), np::dtype::get_builtin<float>());
        for (int j = 0; j < dim; ++j) {
            ((float*)query_vec.get_data())[j] = test_vec_data[query_offset + j];
        }
        std::vector<int> valid_indices;

        for (int j = 0; j < shp1; ++j) {
            int id = idxs_data[idx_offset + j];
            if (id == -1) {
                break;
            }
            valid_indices.push_back(id);
        }

        if (valid_indices.empty()) {
            continue;
        }
        int valid_size = valid_indices.size();
        np::ndarray valid_idx = np::empty(p::make_tuple(valid_size), np::dtype::get_builtin<int>());
        np::ndarray candidate_vec = np::empty(p::make_tuple(valid_size, dim), np::dtype::get_builtin<float>());

        for (int j = 0; j < valid_size; ++j) {
            ((int*)valid_idx.get_data())[j] = valid_indices[j];

            LL train_offset = (LL)valid_indices[j] * dim;
            int candidate_offset = j * dim;
            for (int k = 0; k < dim; ++k) {
                ((float*)candidate_vec.get_data())[candidate_offset + k] = train_vec_data[train_offset + k];
            }
        }
        np::ndarray idx = np::empty(p::make_tuple(topk), np::dtype::get_builtin<int>());
        std::vector<int> sorted_idx = find_k_nearest_neighbors(valid_idx, candidate_vec, query_vec, topk);

        for (int j = 0; j < topk; ++j) {
            ((int*)select_idx.get_data())[i * topk + j] = sorted_idx[j];
        }
    }
    return select_idx;
}

BOOST_PYTHON_MODULE(boost_filter_refine)
{
    Py_Initialize();
    np::initialize();
    p::def("process_mul", &process_mul);
    p::def("Refine", &refine);
}
