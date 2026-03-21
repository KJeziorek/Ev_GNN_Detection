#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <cstring>
#include <torch/extension.h>

class GraphGenerator {
private:
    int width_;
    int height_;
    // Flat arrays for cache-friendly [x * height_ + y] access
    // neigh_matrix_ stores last t at each pixel; -1.0 = empty (assumes t >= 0)
    std::vector<float> neigh_matrix_;
    std::vector<int>   idx_matrix_;
    int global_idx_;

    inline float& neigh(int x, int y) { return neigh_matrix_[x * height_ + y]; }
    inline int&   idx  (int x, int y) { return idx_matrix_  [x * height_ + y]; }

public:
    GraphGenerator(int width, int height)
        : width_(width), height_(height),
          neigh_matrix_(width * height, -1.0f),
          idx_matrix_  (width * height, -1),
          global_idx_(0) {}
    // Returns: (features [N], positions [N,3], edges [E,2])
    // features/positions are float32, edges are int32.
    // Neighbourhood is an ellipsoid: (dx/rx)^2 + (dy/ry)^2 + (dt/rt)^2 <= 1
    // When radius_t <= 0: temporal dimension ignored (pure spatial edges).
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    generate_edges(const torch::Tensor& events,
                   float radius_x, float radius_y, float radius_t) {
        TORCH_CHECK(events.dim() == 2 && events.size(1) == 4,
                    "events must be shape [N, 4]");
        auto ev = events.contiguous();
        if (ev.scalar_type() != torch::kFloat32)
            ev = ev.to(torch::kFloat32);
        auto ev_acc = ev.accessor<float, 2>();
        int num_nodes = ev.size(0);

        int neigh_area = (int)((2 * radius_x + 1) * (2 * radius_y + 1));
        std::vector<int32_t> edges_vec;
        edges_vec.reserve(num_nodes * std::min(neigh_area, 16) * 2);
        std::vector<float> positions_vec;
        positions_vec.reserve(num_nodes * 3);
        std::vector<float> features_vec;
        features_vec.reserve(num_nodes);

        float inv_rx2 = 1.0f / (radius_x * radius_x);
        float inv_ry2 = 1.0f / (radius_y * radius_y);
        bool use_temporal = radius_t > 0.0f;
        float inv_rt2 = use_temporal ? 1.0f / (radius_t * radius_t) : 0.0f;

        int cur_idx = 0;

        for (int i = 0; i < num_nodes; ++i) {
            float xf = ev_acc[i][0];
            float yf = ev_acc[i][1];
            float t  = ev_acc[i][2];
            float p  = ev_acc[i][3];

            int x = (int)xf;
            int y = (int)yf;

            if (t == neigh(x, y)) continue;

            edges_vec.push_back(cur_idx); edges_vec.push_back(cur_idx);
            features_vec.push_back(p);
            positions_vec.push_back(xf); positions_vec.push_back(yf); positions_vec.push_back(t);

            int x_start = std::max(0,           (int)(xf - radius_x));
            int x_end   = std::min(width_  - 1, (int)(xf + radius_x));
            int y_start = std::max(0,           (int)(yf - radius_y));
            int y_end   = std::min(height_ - 1, (int)(yf + radius_y));

            if (use_temporal) {
                for (int j = x_start; j <= x_end; ++j) {
                    float dx  = xf - j;
                    float nx2 = dx * dx * inv_rx2;
                    if (nx2 > 1.0f) continue;

                    const float* nm_row = &neigh_matrix_[j * height_];
                    const int*   im_row = &idx_matrix_  [j * height_];

                    for (int k = y_start; k <= y_end; ++k) {
                        if (nm_row[k] < 0.0f) continue;

                        float dy   = yf - k;
                        float nxy2 = nx2 + dy * dy * inv_ry2;
                        if (nxy2 > 1.0f) continue;

                        float dt = t - nm_row[k];
                        if (nxy2 + dt * dt * inv_rt2 > 1.0f) continue;

                        edges_vec.push_back(cur_idx);
                        edges_vec.push_back(im_row[k]);
                    }
                }
            } else {
                // No temporal filtering — pure spatial ellipse check
                for (int j = x_start; j <= x_end; ++j) {
                    float dx  = xf - j;
                    float nx2 = dx * dx * inv_rx2;
                    if (nx2 > 1.0f) continue;

                    const int* im_row = &idx_matrix_[j * height_];

                    for (int k = y_start; k <= y_end; ++k) {
                        if (im_row[k] < 0) continue;

                        float dy   = yf - k;
                        if (nx2 + dy * dy * inv_ry2 > 1.0f) continue;

                        edges_vec.push_back(cur_idx);
                        edges_vec.push_back(im_row[k]);
                    }
                }
            }

            neigh(x, y) = t;
            idx(x, y)   = global_idx_;
            cur_idx++;
            global_idx_++;
        }

        int N = features_vec.size();
        int E = edges_vec.size() / 2;

        auto features_t  = torch::from_blob(features_vec.data(),  {N, 1},    torch::kFloat32).clone();
        auto positions_t = torch::from_blob(positions_vec.data(), {N, 3}, torch::kFloat32).clone();
        auto edges_t     = torch::from_blob(edges_vec.data(),     {E, 2}, torch::kInt32).clone();

        return std::make_tuple(features_t, positions_t, edges_t);
    }

    void clear() {
        std::fill(neigh_matrix_.begin(), neigh_matrix_.end(), -1.0f);
        // -1 in int32 is all 0xFF bytes, so memset works
        std::memset(idx_matrix_.data(), 0xFF, idx_matrix_.size() * sizeof(int));
        global_idx_ = 0;
    }
};

PYBIND11_MODULE(matrix_neighbour, m) {
    py::class_<GraphGenerator>(m, "GraphGenerator")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("generate_edges", &GraphGenerator::generate_edges,
             "Generate edges from events. Returns (features [N] float32, positions [N,3] float32, edges [E,2] int32). "
             "Neighbourhood is an ellipsoid: (dx/rx)^2 + (dy/ry)^2 + (dt/rt)^2 <= 1. "
             "Set radius_t <= 0 to disable temporal filtering (pure spatial edges).",
             py::arg("events"), py::arg("radius_x"), py::arg("radius_y"), py::arg("radius_t"),
             py::call_guard<py::gil_scoped_release>())
        .def("clear", &GraphGenerator::clear, "Reset matrices to initial state");
}
