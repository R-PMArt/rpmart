"""
Modified from https://github.com/qq456cvb/CPPF/blob/main/models/voting.py
"""
import os
import cupy as cp


helper_math_path = os.path.join(os.path.dirname(__file__), 'helper_math.cuh')


ppf_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting(
        const float *points, const float *outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }
''', 'ppf_voting')

ppf_retrieval_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting_retrieval(
        const float *point_pairs, const float *outputs, const float *probs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            float3 a = make_float3(point_pairs[idx * 6], point_pairs[idx * 6 + 1], point_pairs[idx * 6 + 2]);
            float3 b = make_float3(point_pairs[idx * 6 + 3], point_pairs[idx * 6 + 4], point_pairs[idx * 6 + 5]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
                atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
                atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
            }
        }
    }
''', 'ppf_voting_retrieval')

ppf_direct_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf_voting_direct(
        const float *points, const float *outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int grid_x, int grid_y, int grid_z
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            int a_idx = point_idxs[idx];
            float3 c = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float prob = probs[idx];
            float3 offset = make_float3(outputs[idx * 3], outputs[idx * 3 + 1], outputs[idx * 3 + 2]);
            float3 center_grid = (c - offset - make_float3(corner[0], corner[1], corner[2])) / res;
            if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                return;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            float lll = w0.x * w0.y * w0.z;
            float llh = w0.x * w0.y * w1.z;
            float lhl = w0.x * w1.y * w0.z;
            float lhh = w0.x * w1.y * w1.z;
            float hll = w1.x * w0.y * w0.z;
            float hlh = w1.x * w0.y * w1.z;
            float hhl = w1.x * w1.y * w0.z;
            float hhh = w1.x * w1.y * w1.z;

            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], lll * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], llh * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], lhl * prob);
            atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], lhh * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_floor.z], hll * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_floor.y * grid_z + center_grid_ceil.z], hlh * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_floor.z], hhl * prob);
            atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z + center_grid_ceil.y * grid_z + center_grid_ceil.z], hhh * prob);
        }
    }
''', 'ppf_voting_direct')


rot_voting_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting(
        const float *points, const float *preds_rot, float3 *outputs_up, const int *point_idxs,
        int n_ppfs, int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx];
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= length(ab);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                up = up / (length(up) + 1e-7);
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting')

rot_voting_retrieval_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void rot_voting_retrieval(
        const float *point_pairs, const float *preds_rot, float3 *outputs_up, 
        int n_ppfs, int n_rots
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float rot = preds_rot[idx];
            float3 a = make_float3(point_pairs[idx * 6], point_pairs[idx * 6 + 1], point_pairs[idx * 6 + 2]);
            float3 b = make_float3(point_pairs[idx * 6 + 3], point_pairs[idx * 6 + 4], point_pairs[idx * 6 + 5]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= length(ab);

            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7);
            float3 y = cross(x, ab);
            
            for (int i = 0; i < n_rots; i++) {
                float angle = i * 2 * M_PI / n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 up = tan(rot) * offset + (tan(rot) > 0 ? ab : -ab);
                up = up / (length(up) + 1e-7);
                outputs_up[idx * n_rots + i] = up;
            }
        }
    }
''', 'rot_voting_retrieval')


ppf4d_kernel = cp.RawKernel(f'#include "{helper_math_path}"\n' + r'''
    #define M_PI 3.14159265358979323846264338327950288
    extern "C" __global__
    void ppf4d_voting(
        const float *points, const float *outputs, const float *rot_outputs, const float *probs, const int *point_idxs, float *grid_obj, const float *corner, const float res,
        int n_ppfs, int n_rots, int grid_x, int grid_y, int grid_z, int grid_w
    ) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n_ppfs) {
            float proj_len = outputs[idx * 2];
            float odist = outputs[idx * 2 + 1];
            if (odist < res) return;
            int a_idx = point_idxs[idx * 2];
            int b_idx = point_idxs[idx * 2 + 1];
            float3 a = make_float3(points[a_idx * 3], points[a_idx * 3 + 1], points[a_idx * 3 + 2]);
            float3 b = make_float3(points[b_idx * 3], points[b_idx * 3 + 1], points[b_idx * 3 + 2]);
            float3 ab = a - b;
            if (length(ab) < 1e-7) return;
            ab /= (length(ab) + 1e-7);
            float3 c = a - ab * proj_len;

            // float prob = max(probs[a_idx], probs[b_idx]);
            float prob = probs[idx];
            float3 co = make_float3(0.f, -ab.z, ab.y);
            if (length(co) < 1e-7) co = make_float3(-ab.y, ab.x, 0.f);
            float3 x = co / (length(co) + 1e-7) * odist;
            float3 y = cross(x, ab);
            int adaptive_n_rots = min(int(odist / res * (2 * M_PI)), n_rots);
            // int adaptive_n_rots = n_rots;
            for (int i = 0; i < adaptive_n_rots; i++) {
                float angle = i * 2 * M_PI / adaptive_n_rots;
                float3 offset = cos(angle) * x + sin(angle) * y;
                float3 center_grid = (c + offset - make_float3(corner[0], corner[1], corner[2])) / res;
                if (center_grid.x < 0.01 || center_grid.y < 0.01 || center_grid.z < 0.01 || 
                    center_grid.x >= grid_x - 1.01 || center_grid.y >= grid_y - 1.01 || center_grid.z >= grid_z - 1.01) {
                    continue;
                }
                int3 center_grid_floor = make_int3(center_grid);
                int3 center_grid_ceil = center_grid_floor + 1;
                float3 residual = fracf(center_grid);
                
                float3 w0 = 1.f - residual;
                float3 w1 = residual;
                
                float lll = w0.x * w0.y * w0.z;
                float llh = w0.x * w0.y * w1.z;
                float lhl = w0.x * w1.y * w0.z;
                float lhh = w0.x * w1.y * w1.z;
                float hll = w1.x * w0.y * w0.z;
                float hlh = w1.x * w0.y * w1.z;
                float hhl = w1.x * w1.y * w0.z;
                float hhh = w1.x * w1.y * w1.z;

                for (int j = 0; j < grid_w; j++) {
                    atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z * grid_w + center_grid_floor.y * grid_z * grid_w + center_grid_floor.z * grid_w + j], rot_outputs[idx * grid_w + j] * lll * prob);
                    atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z * grid_w + center_grid_floor.y * grid_z * grid_w + center_grid_ceil.z * grid_w + j], rot_outputs[idx * grid_w + j] * llh * prob);
                    atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z * grid_w + center_grid_ceil.y * grid_z * grid_w + center_grid_floor.z * grid_w + j], rot_outputs[idx * grid_w + j] * lhl * prob);
                    atomicAdd(&grid_obj[center_grid_floor.x * grid_y * grid_z * grid_w + center_grid_ceil.y * grid_z * grid_w + center_grid_ceil.z * grid_w + j], rot_outputs[idx * grid_w + j] * lhh * prob);
                    atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z * grid_w + center_grid_floor.y * grid_z * grid_w + center_grid_floor.z * grid_w + j], rot_outputs[idx * grid_w + j] * hll * prob);
                    atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z * grid_w + center_grid_floor.y * grid_z * grid_w + center_grid_ceil.z * grid_w + j], rot_outputs[idx * grid_w + j] * hlh * prob);
                    atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z * grid_w + center_grid_ceil.y * grid_z * grid_w + center_grid_floor.z * grid_w + j], rot_outputs[idx * grid_w + j] * hhl * prob);
                    atomicAdd(&grid_obj[center_grid_ceil.x * grid_y * grid_z * grid_w + center_grid_ceil.y * grid_z * grid_w + center_grid_ceil.z * grid_w + j], rot_outputs[idx * grid_w + j] * hhh * prob);
                }
            }
        }
    }
''', 'ppf4d_voting')
