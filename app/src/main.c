#include <hal_types.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include "common.h"
#include "hif_config.h"
#include "hif.h"
#include "hif_com.h"
#include "mmw_ctrl.h"
#include "mmw_alg_pointcloud.h"
#include "mmw_point_cloud_psic_lib.h"
#include "mmw_app_pointcloud.h"
#include "mmw_point_cloud_ut.h"
#include "mmw_alg_debug.h"
#include "mmw_app_pointcloud_config.h"
#include "mmw_report.h"
#include "mmw_type.h"
#include "ll_utils.h"
#include "mmw_ctrl_os_api.h"

bool switch_to_point = false; // 是否切换至点云状态(离床)
bool switch_to_range = false; // 是否切换至fft状态（上床）

// ================== 全局配置 ==================

// 床的空间范围
//double range_array[3][2] = {{-60, 60}, {20, 200}, {-95, 30}};
double range_array[3][2] = {{-60, 60}, {30, 200}, {-75, 60}};

// ===== 全局点云统计数组 =====
#define FRAME_BUFFER_SIZE 400 // 测试30，实际使用的时候300，一分钟上报两次
#define MAX_POINT_NUM 100
#define RANGE_BIN_NUM 15

// 定义各种状态
#define STATUS_NONE 0 // 床上有人，包括环境内无人或者框外有人
#define STATUS_ON_BED 1 // 上床
#define STATUS_ACTIVE 2 // 床上活动
#define STATUS_TRANSITION 3 // 过渡状态
#define STATUS_SLEEP 4 // 睡眠
#define STATUS_OFF_BED 5 // 离床

static int point_num_array[FRAME_BUFFER_SIZE];
static int inside_array[FRAME_BUFFER_SIZE];
static int outside_array[FRAME_BUFFER_SIZE];
static int frame_count = 0;

#define SLEEP_WINDOW 3  // 连续帧判定窗口
//#define SLEEP_WINDOW 6  // 连续帧判定窗口

/*-----------Range FFT 需要使用的---------------*/
// 阈值，可根据实际调试调整
//#define OVERALL_VAR_SLEEP 600
//#define BIN_VAR_SLEEP 900
//#define ENTROPY_SLEEP 250 // 原值 2.8f
#define OVERALL_VAR_SLEEP 1000
#define BIN_VAR_SLEEP 600
#define ENTROPY_SLEEP 280 

//#define OVERALL_VAR_WAKE 900
//#define BIN_VAR_WAKE 1200
//#define ENTROPY_WAKE 250
#define OVERALL_VAR_WAKE 1400
#define BIN_VAR_WAKE 1000
#define ENTROPY_WAKE 250
#define OFF_BED_THRESHOLD 50 

// 全局状态机变量
static int g_state = STATUS_ACTIVE;
static int g_sleep_count = 0;
static int g_wake_count = 0;
/*---------------------------------------------*/

// 全局存储获取到的range fft数据，包括实部和虚部
Complex16_RealImag buffer[RANGE_BIN_NUM + 1] __aligned(4);
int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM] = {0};
int16_t imag_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM] = {0};

typedef struct{
    int state;
    int bpm;
}SleepStatusResult;

// 统计一帧的 inside/outside 点数
void count_points_in_frame(Detection3D_Data *data, int *total, int *inside, int *outside)
{
    *total = 0;
    *inside = 0;
    *outside = 0;
	
    if (data && data->motion_points && data->motion_points_num > 0) {
        PointCloud_Cart *points = (PointCloud_Cart *)data->motion_points;

        // 修复：如果点数小于 MAX_POINT_NUM，取实际点数，而不是 0
        *total = (data->motion_points_num > MAX_POINT_NUM) ? MAX_POINT_NUM : data->motion_points_num;

        for (int i = 0; i < *total; i++) {
            int x_in_range = points[i].x >= range_array[0][0] && points[i].x <= range_array[0][1];
            int y_in_range = points[i].y >= range_array[1][0] && points[i].y <= range_array[1][1];
            int z_in_range = points[i].z >= range_array[2][0] && points[i].z <= range_array[2][1];
            if (x_in_range && y_in_range && z_in_range) {
                (*inside)++;
            } else {
                (*outside)++;
            }
        }
    }
//	printf("[Frame %d] total=%d inside=%d outside=%d\n", frame_count, *total, *inside, *outside);

}

// 统计 600 帧后的状态
int judge_bed_status(int *point_nums, int *inside_nums, int *outside_nums,
                     int num_frames, int inside_th, int none_th, int outside_th)
{
    // 找到最大点数对应的 index
    int max_points = 0;
    int best_idx = -1;
    for (int i = 0; i < num_frames; i++) {
        if (point_nums[i] >= max_points) { // 尽量判定靠后时间的点云
            max_points = point_nums[i];
            best_idx = i;
        }
    }

    if (best_idx < 0) {
        return STATUS_NONE; // 默认无人
    }

    int inside_point_num = inside_nums[best_idx];
    int outside_point_num = outside_nums[best_idx];

    // DEBUG 信息
//	printf("[DEBUG] MaxIdx=%d, MaxPoints=%d, Inside=%d, Outside=%d\n", best_idx, max_points, inside_point_num, outside_point_num);

    // 判定逻辑
    if (inside_point_num >= inside_th && inside_point_num >= outside_point_num) {
        return STATUS_ON_BED;   // 上床
    } else if (max_points <= none_th) {
        return STATUS_NONE;   // 无人
    } else if (outside_point_num >= outside_th && outside_point_num > inside_point_num) {
        return STATUS_NONE;   // 框外有人
    }

    return STATUS_NONE; // 默认无人
}

// ================= Range-FFT 处理函数 ==================
// 计算一维数组的标准差
static float calc_variance(const int16_t *data, int length)
{
    float mean = 0;
    for (int i = 0; i < length; i++)
        mean += data[i];
    mean /= length;

    float var = 0;
    for (int i = 0; i < length; i++)
        var += (data[i] - mean) * (data[i] - mean);
    var /= length;

    return sqrtf(var);
}
    
static int16_t calc_entropy(const int16_t *data, int length, int num_bins) { 
    int hist[30] = {0}; // 默认30个bin 
    int16_t min = data[0], max = data[0]; 
    for (int i = 1; i < length; i++) { 
        if (data[i] < min) min = data[i]; 
        if (data[i] > max) max = data[i]; 
    } 
    float range = max - min + 1e-6f; 
	// 把 [min, max] 区间分成 num_bins 份
	// hist[i] 统计第 i 个 bin 的样本数量
    for (int i = 0; i < length; i++) { 
        int idx = (int)((data[i] - min) / range * num_bins); 
        if (idx >= num_bins) idx = num_bins - 1; hist[idx]++; 
    } 
	// 计算熵值
    float entropy = 0; 
    for (int i = 0; i < num_bins; i++) { 
        if (hist[i] == 0) continue; 
        float p = (float)hist[i] / length; //概率Pi
//        entropy -= p * simple_logf(p); 
		// 新版有ln函数
		entropy -= p * logf(p);
    } 
	return (int16_t)(entropy * 100); 
}
// 计算二维矩阵的全局方差
static float calc_overall_variance(const int16_t data[][RANGE_BIN_NUM], int num_frames, int num_bins) {
    int total_elements = num_frames * num_bins;
    float mean = 0;
    for (int f = 0; f < num_frames; f++) {
        for (int b = 0; b < num_bins; b++) {
            mean += data[f][b];
        }
    }
    mean /= total_elements;

    float variance = 0;
    for (int f = 0; f < num_frames; f++) {
        for (int b = 0; b < num_bins; b++) {
            variance += (data[f][b] - mean) * (data[f][b] - mean);
        }
    }
    variance /= total_elements;

//    return sqrtf(variance);
	return variance;
}

// ================= Range-FFT 处理函数 ==================

// static 全局，保证不会越界
static float bin_vars[RANGE_BIN_NUM];
static float bin_entropies[RANGE_BIN_NUM];
static int16_t bin_data[FRAME_BUFFER_SIZE];
// static float all_values[FRAME_BUFFER_SIZE * RANGE_BIN_NUM];

// 计算 overall_var, max_var, median_entropy
static void compute_var_entropy_time(int16_t data[][RANGE_BIN_NUM], int num_frames,
                                    float *overall_var, float *max_var, int16_t *median_entropy)
{
    int idx = 0;
    for (int b = 0; b < RANGE_BIN_NUM; b++) {
        for (int f = 0; f < num_frames; f++) {
			// 对于每一个距离 bin（即距离位置），取出它在所有帧上的时间序列信号
            bin_data[f] = data[f][b];
        }
		// 计算该bin的方差和熵
        bin_vars[b] = calc_variance(bin_data, num_frames);
        bin_entropies[b] = calc_entropy(bin_data, num_frames, 30);
    }

    // 使用中间bin的值计算 overall_var
//    *overall_var = bin_vars[RANGE_BIN_NUM / 2];
	*overall_var = calc_overall_variance(data, num_frames, RANGE_BIN_NUM);

    // max_var最大bin方差，判断是否离床
    *max_var = bin_vars[0];
    for (int i = 1; i < RANGE_BIN_NUM; i++) {
        if (bin_vars[i] > *max_var)
            *max_var = bin_vars[i];
    }

    // median_entropy
	// 冒泡排序取中间值
    for (int i = 0; i < RANGE_BIN_NUM - 1; i++) {
        for (int j = 0; j < RANGE_BIN_NUM - i - 1; j++) {
            if (bin_entropies[j] > bin_entropies[j + 1]) {
                float tmp = bin_entropies[j];
                bin_entropies[j] = bin_entropies[j + 1];
                bin_entropies[j + 1] = tmp;
            }
        }
    }
    *median_entropy = (int16_t)(bin_entropies[(int16_t)(RANGE_BIN_NUM / 2)]);
//    printf("[DEBUG] overall_var=%d, max_var=%d, median_entropy*100 = %d\n", (int)*overall_var, (int)*max_var, (int)*median_entropy);
}

// TODO 获取呼吸相关的函数
// 目前存在的问题主要是获取相位的函数实现起来相对比较困难，需要用近似计算来替代
#define FS  20.0f // 采样率 Hz
#ifndef M_PI
#define M_PI 3.14159f
#endif

static float unwrap_phase(float prev, float current) {
    float diff = current - prev;

    if (diff > M_PI) {
        diff -= 2.0f * M_PI;
    } else if (diff < -M_PI) {
        diff += 2.0f * M_PI;
    }

    return prev + diff;
}

//static float sig_padded[FRAME_BUFFER_SIZE + 100];
//
//// 双窗口滤波
//static void double_window_filter(float* phase_unwrapped, int n)
//{
//	int outer_win_size = 21;
//	int inner_win_size = 3;
//	float sigma_d = 5.0;
//	float sigma_r = 0.5;
//    int half_outer = outer_win_size / 2;
//    int half_inner = inner_win_size / 2;
//    int pad = half_outer + half_inner;
//    int padded_len = n + 2 * pad;
//
//    // 边界扩展
//    for (int i = 0; i < pad; i++)
//        sig_padded[i] = phase_unwrapped[0];
//    for (int i = 0; i < n; i++)
//        sig_padded[i + pad] = phase_unwrapped[i];
//    for (int i = 0; i < pad; i++)
//        sig_padded[n + pad + i] = phase_unwrapped[n - 1];
//
//    // 双窗口滤波
//    for (int i = 0; i < n; i++) {
//        int i_center = i + pad;
//        int outer_start = i_center - half_outer;
//        int outer_end = i_center + half_outer;
//
//        float total_weight = 0.0;
//        float weighted_sum = 0.0;
//
//        for (int j = outer_start; j <= outer_end; j++) {
//            // 空间权重
//            float dist_spatial = fabs((float)(i_center - j));
//            float w_d = exp(-(dist_spatial * dist_spatial) / (2.0 * sigma_d * sigma_d));
//
//            // 值域权重（基于内窗口）
//            float dist_range = 0.0;
//            for (int k = -half_inner; k <= half_inner; k++) {
//                float inner_i = sig_padded[i_center + k];
//                float inner_j = sig_padded[j + k];
//                dist_range += fabs(inner_i - inner_j);
//            }
//            float w_r = exp(-(dist_range * dist_range) / (2.0 * sigma_r * sigma_r));
//
//            float weight = w_d * w_r;
//            total_weight += weight;
//            weighted_sum += weight * sig_padded[j];
//        }
//
//        if (total_weight > 0.0)
//            phase_unwrapped[i] = (float)(weighted_sum / total_weight);
//    }
//}

static float sig_buffer[FRAME_BUFFER_SIZE + 200];   
static float tmp_buffer[FRAME_BUFFER_SIZE + 200];

static void moving_average(const float* sig, float* out, int n, int window)
{
    if (window <= 1) {
        for (int i = 0; i < n; i++) out[i] = sig[i];
        return;
    }

    int right = window - 1;
    if (right >= n) right = n - 1;

    float s = 0.0f;
    for (int i = 0; i <= right; i++)
        s += sig[i];

    int left = 0;

    for (int i = 0; i < n; i++)
    {
        int length = right - left + 1;
        out[i] = s / (float)length;

        // slide
        s -= sig[left];
        left++;
        right++;

        if (right < n)
            s += sig[right];
        else
            s += sig[n - 1];   // edge padding
    }
}

static void remove_bashline_drift(float* phase_unwrapped, int n)
{
    const float fs = 20.0f;
    const float win_len_big   = 5.0f;   // 大窗口
    const float win_len_small = 1.7f;   // 小窗口

    int win_big   = (int)(fs * win_len_big);
    int win_small = (int)(fs * win_len_small);

    if (win_big > n)   win_big = n;
    if (win_small > n) win_small = n;

    // 1. 大窗口求 baseline：tmp_buffer
    moving_average(phase_unwrapped, tmp_buffer, n, win_big);

    // corrected = signal - baseline → 放入 sig_buffer
    for (int i = 0; i < n; i++)
        sig_buffer[i] = phase_unwrapped[i] - tmp_buffer[i];

    // 2. 小窗口平滑：输出直接写回 phase_unwrapped
    moving_average(sig_buffer, phase_unwrapped, n, win_small);
}

// butterworth带通滤波
#define FILTER_ORDER 8

//static float b[FILTER_ORDER + 1] = {
//    1.32937289e-05f,  0.00000000e+00f, -5.31749156e-05f,  0.00000000e+00f,
//    7.97623734e-05f,  0.00000000e+00f, -5.31749156e-05f,  0.00000000e+00f,
//    1.32937289e-05f
//};

static float b[FILTER_ORDER + 1] = {
    3.12389769e-05f,  
    0.00000000e+00f,  
    -1.24955908e-04f, 
    0.00000000e+00f,  
    1.87433862e-04f,  
    0.00000000e+00f,  
    -1.24955908e-04f, 
    0.00000000e+00f, 
    3.12389769e-05f  
};

//static float a[FILTER_ORDER + 1] = {
//    1.00000000f,  -7.65278270f,  25.64597452f, -49.15718723f,
//    58.94461344f, -45.27863033f,  21.75890323f,  -5.98080126f,
//    0.71991033f
//};

static float a[FILTER_ORDER + 1] = {
    1.0f,          
    -7.56722638f,   
    25.08216965f,  
    -47.56384375f,  
    56.44130747f,   
    -42.91716792f,  
    20.42130903f,  
    -5.55955857f,   
    0.66301048f     
};

float x_history[FILTER_ORDER] = {0};
float y_history[FILTER_ORDER] = {0};
int filter_initialized = 0;

// 初始化滤波器（重置历史数据）
static void init_filter(void) {
    memset(x_history, 0, FILTER_ORDER * sizeof(float));
    memset(y_history, 0, FILTER_ORDER * sizeof(float));
    filter_initialized = 1;
}

// 对单个样本进行滤波
static float filter_sample(float x_n) {
    if (!filter_initialized) {
        init_filter();
    }
    
    float acc = b[0] * x_n;

    // 累加历史项
    for (int i = 1; i <= FILTER_ORDER; i++) {
        acc += b[i] * x_history[i - 1];
        acc -= a[i] * y_history[i - 1];
    }

    // 更新历史缓存 - 分别更新避免数据覆盖
    for (int i = FILTER_ORDER - 1; i > 0; i--) {
        x_history[i] = x_history[i - 1];
    }
    for (int i = FILTER_ORDER - 1; i > 0; i--) {
        y_history[i] = y_history[i - 1];
    }
    
    x_history[0] = x_n;
    y_history[0] = acc;

    return acc;
}

// 对整个数组进行滤波
static void filter_array(float *x, float *y, int n) {
	init_filter();
    
    for (int i = 0; i < n; i++) {
        y[i] = filter_sample(x[i]);
    }
}

#define MIN_PEAK_DISTANCE (int)(FS * 0.6)  // 峰间最小距离（0.6秒）
static int peaks[512];
//
//// 检测峰值
//static int find_peaks(const float *signal, int length, int *peaks, int max_peaks) {
//    int count = 0;
//    int last_peak = -MIN_PEAK_DISTANCE;
//
//    for (int i = 1; i < length - 1; i++) {
//        // 简单的局部极大值检测
//        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
//            // 距离约束：相邻峰至少间隔 MIN_PEAK_DISTANCE
//            if (i - last_peak >= MIN_PEAK_DISTANCE) {
//                if (count < max_peaks) {
//                    peaks[count++] = i;
//                    last_peak = i;
//                }
//            }
//        }
//    }
//
//    return count;  // 返回检测到的峰数量
//}

static int find_peaks_improved(const float *signal, int length, int *peaks, int max_peaks) {
    int count = 0;
    int window = FS / 2;  // 0.5秒窗口
    
    for (int i = window; i < length - window; i++) {
        // 检查是否为窗口内的最大值
        int is_peak = 1;
        float max_val = signal[i];
        
        for (int j = i - window; j <= i + window; j++) {
            if (j == i) continue;
            if (signal[j] >= max_val) {
                is_peak = 0;
                break;
            }
        }
        
        if (is_peak) {
            // 距离检查
            int valid = 1;
            for (int p = 0; p < count; p++) {
                if (abs(i - peaks[p]) < MIN_PEAK_DISTANCE) {
                    // 如果找到更近的峰值，保留更高的那个
                    if (signal[i] > signal[peaks[p]]) {
                        peaks[p] = i;  // 替换为更高的峰值
                    }
                    valid = 0;
                    break;
                }
            }
            
            if (valid && count < max_peaks) {
                peaks[count++] = i;
            }
        }
    }
    
    return count;
}

// 呼吸率计算
static float compute_breath_rate(const float *signal, int length) {
    int num_peaks = find_peaks_improved(signal, length, peaks, 512);
	printf("num_peaks:%d\n",num_peaks);
	for(int i=0;i<num_peaks;i++){
		printf("%d ",peaks[i]);
	}
	printf("\n");

    if (num_peaks < 2) return 0.0; // 峰太少，无法计算

    float total_period = 0.0;
    for (int i = 1; i < num_peaks; i++) {
        int diff = peaks[i] - peaks[i - 1];
        total_period += (float)diff / FS;
    }

    float mean_period = total_period / (num_peaks - 1);
    if (mean_period <= 0.0) return 0.0;

    float bpm = 60.0 / mean_period;  // 呼吸率（次/分钟）
    return bpm;
}

static float compute_mean(float* phase)
{
	float mean = 0.0f;
	for (int t = 0; t < FRAME_BUFFER_SIZE; t++) mean += phase[t];
	mean /= (float)FRAME_BUFFER_SIZE;
	return mean;
}

static float phase_tmp[FRAME_BUFFER_SIZE] = {0};
static float phase_sequences[RANGE_BIN_NUM][FRAME_BUFFER_SIZE] = {0};
static float energies[RANGE_BIN_NUM] = {0};
static float auto_vals[RANGE_BIN_NUM] = {0};
static float scores[RANGE_BIN_NUM] = {0};
static float ac_pos[FRAME_BUFFER_SIZE] = {0};

static int choose_bin(int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM],
				int16_t imag_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM])
{
	float w_auto = 0.6f;     // 周期性最重要
    float w_energy = 0.4f;   // 能量越大越好
    float max_energy = 0.0f;
    float max_auto = 0.0f;
	
	// 计算能量
	for (int b = 0; b < RANGE_BIN_NUM; b++) {
		// 计算每帧的相位并解缠
        float prev = atan2f((float)imag_data[0][b], (float)real_data[0][b]);
        phase_tmp[0] = prev;
        for (int t = 1; t < FRAME_BUFFER_SIZE; t++) {
            float raw = atan2f((float)imag_data[t][b], (float)real_data[t][b]);
            phase_tmp[t] = unwrap_phase(prev, raw);
            prev = phase_tmp[t];
        }

//		double_window_filter(phase_tmp, FRAME_BUFFER_SIZE);
		remove_bashline_drift(phase_tmp, FRAME_BUFFER_SIZE);
	
		// 仅用于自相关，不需要那么精确，不用滤波，缩短执行时间			

		// 去直流，归一化
		float mean1 = compute_mean(phase_tmp);
		for (int t = 0; t < FRAME_BUFFER_SIZE; t++){
			phase_tmp[t] -= mean1;
			phase_sequences[b][t] = phase_tmp[t];
		}
			
		// 计算信号能量
		float energy = 0.0f;
		for (int t = 0; t < FRAME_BUFFER_SIZE; t++) {
			float real_val = (float)real_data[t][b];
			float imag_val = (float)imag_data[t][b];
			energy += real_val * real_val + imag_val * imag_val;
		}
		energy /= (float)FRAME_BUFFER_SIZE;
		energies[b] = energy;
			
		if (energy > max_energy) {
			max_energy = energy;
		}
	}
	for (int b = 0; b < RANGE_BIN_NUM; b++) {
		energies[b] = (max_energy > 1e-12f) ? energies[b] / max_energy : 0.0f;
	}
	
	
	// 计算自相关评估周期性
	for (int b = 0; b < RANGE_BIN_NUM; b++) {
		int N = FRAME_BUFFER_SIZE;
		
		// 先计算均值
		float x_mean = compute_mean(phase_sequences[b]);
		
		// 计算自相关（非中心化方式，与Python代码一致）
		for (int lag = 0; lag < N; lag++) {
			double sum = 0.0;
			int limit = N - lag;
			for (int n = 0; n < limit; n++) {
				double x1 = phase_sequences[b][n] - x_mean;
				double x2 = phase_sequences[b][n + lag] - x_mean;
				sum += x1 * x2;
			}
			ac_pos[lag] = (float)sum;
		}
		
		int peak_count = find_peaks_improved(ac_pos, N, peaks, 512);
		if (peak_count < 2) {
			auto_vals[b] = 0.0f;   
		} else {
			int idx = peaks[1];     
			auto_vals[b] = ac_pos[idx];
		}
		if (auto_vals[b] > max_auto) {
			max_auto = auto_vals[b];
		}
	}

	for (int b = 0; b < RANGE_BIN_NUM; b++) {
		auto_vals[b] = (max_auto > 1e-12f) ? auto_vals[b] / max_auto : 0.0f;
	}
	
	// 选择最佳 bin
    int best_bin = 0;
    float max_score = -1.0f;
    for (int b = 0; b < RANGE_BIN_NUM; b++) {
        scores[b] = w_auto * auto_vals[b] + w_energy * energies[b];
        
//        printf("bin %d: auto_norm=%.2f energy=%.2f score=%.2f\n", 
//               b, auto_vals[b], energies[b], scores[b]);
        
        if (scores[b] > max_score) {
            max_score = scores[b];
            best_bin = b;
        }
    }
    printf("best_bin = %d\n", best_bin);
	return best_bin;
}


static float phase[FRAME_BUFFER_SIZE];
static float phase_filtered[FRAME_BUFFER_SIZE];
// 主函数：计算呼吸率 (bpm)
static float estimate_respiration_rate(int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM],
                                int16_t imag_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM])
{
	
	// 1. 选择bin（只做预处理，不做完整滤波）
    int best_bin = choose_bin(real_data, imag_data);
	
	// 2. 提取目标bin的原始相位
	float prev_phase = atan2f((float)imag_data[0][best_bin], (float)real_data[0][best_bin]);
	phase[0] = prev_phase;
    for (int t = 1; t < FRAME_BUFFER_SIZE; t++) {
        float raw = atan2f((float)imag_data[t][best_bin], (float)real_data[t][best_bin]);
        phase[t] = unwrap_phase(prev_phase, raw);
        prev_phase = phase[t];
    }
	
	// 3. 滤波
//	double_window_filter(phase, FRAME_BUFFER_SIZE);
	remove_bashline_drift(phase, FRAME_BUFFER_SIZE);
	
	// 4. 带通滤波
	filter_array(phase, phase_filtered, FRAME_BUFFER_SIZE);
	
    // 5. 计算呼吸率
    float bpm = compute_breath_rate(phase_filtered, FRAME_BUFFER_SIZE);
	
    return bpm;
}
//---------------------------------------------------------

// ===================== 主判定函数 =====================
SleepStatusResult judge_sleep_status(int16_t real_data[][RANGE_BIN_NUM], int16_t imag_data[][RANGE_BIN_NUM], int num_frames)
{
    SleepStatusResult result = {0};
    float overall_var, max_var;
    int bpm = 0;
    int16_t median_entropy;

    // 1. 计算统计指标，用于睡眠状态的特征提取
	// overall_var衡量整体波动性，判断活动和睡眠
	// max_var衡量单个bin的波动性，判断是否有人离床
	// median_entropy衡量信号的规律性，睡眠时呼吸规律
    compute_var_entropy_time(real_data, num_frames, &overall_var, &max_var, &median_entropy);

    // 2. 离床判断（优先级最高）：无人
    if (max_var < OFF_BED_THRESHOLD) {  // 对应Python中的mv < 50
        g_state = STATUS_OFF_BED; // 离床/无人
		// 切换至点云模式
        g_sleep_count = 0;
        g_wake_count = 0;
        result.state = g_state;
        result.bpm = bpm;
        return result;
    }

    // 3. 状态机逻辑
    int score_sleep = 0, score_wake = 0;
    if (overall_var <= OVERALL_VAR_SLEEP) score_sleep++;
    if (max_var <= BIN_VAR_SLEEP) score_sleep++;
    if (median_entropy >= ENTROPY_SLEEP) score_sleep++;

    if (overall_var > OVERALL_VAR_WAKE) score_wake++;
    if (max_var > BIN_VAR_WAKE) score_wake++;
    if (median_entropy < ENTROPY_WAKE) score_wake++;

    switch (g_state) {
        case STATUS_ACTIVE:
            if (score_sleep >= 2) {
                g_sleep_count++;
                if (g_sleep_count >= SLEEP_WINDOW) {
                    g_state = STATUS_SLEEP;
                    g_sleep_count = 0;
                } else {
                    g_state = STATUS_TRANSITION;
                }
            } else {
                g_sleep_count = 0;
                g_state = STATUS_ACTIVE;
            }
            break;
        case STATUS_SLEEP:
            if (score_wake >= 2) {
                g_wake_count++;
                if (g_wake_count >= SLEEP_WINDOW) {
                    g_state = STATUS_ACTIVE;
                    g_wake_count = 0;
                } else {
                    g_state = STATUS_TRANSITION;
                }
            } else {
                g_wake_count = 0;
				// 睡眠时可尝试估计呼吸率
				bpm = estimate_respiration_rate(real_data, imag_data);
            }
            break;
        case STATUS_TRANSITION:
            if (score_sleep >= 2) {
                g_sleep_count++;
                if (g_sleep_count >= SLEEP_WINDOW) {
                    g_state = STATUS_SLEEP;
                    bpm = estimate_respiration_rate(real_data, imag_data);
                    g_sleep_count = 0;
                }
            } else if (score_wake >= 2) {
                g_wake_count++;
                if (g_wake_count >= SLEEP_WINDOW) {
                    g_state = STATUS_ACTIVE;
                    g_wake_count = 0;
                }
            }
            break;
        default:
            g_state = STATUS_ACTIVE;
            g_sleep_count = 0;
            g_wake_count = 0;
            break;
    }


    result.state = g_state;
    result.bpm = bpm;
    return result;
}

// ================== 回调：Range FFT 数据处理回调 ==================
static void mmw_data_process(void *mmw_data) {
    uint32_t data_status = mmw_data_get_status();
    int ret;

    if (data_status & BIT(MMW_DATA_TYPE_1DFFT)) {
        // 读取 Range FFT 数据
		// 天线5，从0开始
		// 1 0：
        int tx = 0, rx = 0;
        ret = mmw_fft_data(buffer, RANGE_BIN_NUM, tx, rx, 12);
        if (ret) {
            printf("mmw_fft_data error! %d\n", ret);
            return;
        }

        // 实际能用的点数 = min(RANGE_BIN_NUM, ret)
		// real_data[num_frames][RANGE_BIN_NUM] 二维矩阵：时间帧 × 距离单元
        if (frame_count < FRAME_BUFFER_SIZE) {
//            for (int i = 1; i < RANGE_BIN_NUM + 1; i++) {
//                real_data[frame_count][i] = buffer[i].real;  // 保持 +1 逻辑
//                imag_data[frame_count][i] = buffer[i].imag;
//            }
			for (int i = 0; i < RANGE_BIN_NUM; i++) {
				real_data[frame_count][i] = buffer[i + 1].real;
				imag_data[frame_count][i] = buffer[i + 1].imag;
			}
        }

        if (frame_count >= FRAME_BUFFER_SIZE) {
			printf("Range FFT Data Collection Full!\n");
            // 收满后判定睡眠状态
            SleepStatusResult sleep_status = judge_sleep_status(real_data, imag_data, FRAME_BUFFER_SIZE);
            if (sleep_status.state == STATUS_OFF_BED) {
//				printf("Sleep Status: Off bed\n");
				printf("%d\n", sleep_status.state);
                // 离床：切回点云模式
                switch_to_point = true;
                int ret = mmw_ctrl_stop();
                if (ret) {
                    printf("mmw_ctrl_stop error: %d\n", ret);
                }
                g_state = STATUS_ON_BED; // 重置状态机
                frame_count = 0;
            } else if (sleep_status.state == STATUS_SLEEP) {
                printf("%d\t%d\n", sleep_status.state, sleep_status.bpm);
            } else {
                printf("%d\n", sleep_status.state);
            }
            frame_count = 0;
        } else {
            frame_count++;
        }
    }
}

// ================== 回调：点云检测回调 ==================
static int detection3d_cpuf_cb(Detection3D_Data *data, Detection3D_State *state, void *arg)
{
    // 统计当前帧的 inside/outside 点数
    int total, inside, outside;
    count_points_in_frame(data, &total, &inside, &outside);

    if (frame_count < FRAME_BUFFER_SIZE) {
        point_num_array[frame_count] = total;
        inside_array[frame_count] = inside;
        outside_array[frame_count] = outside;
        frame_count++;
    }

    // 当收满 600 帧，执行一次统计
    if (frame_count >= FRAME_BUFFER_SIZE) {
        int result = judge_bed_status(point_num_array, inside_array, outside_array,
                                      FRAME_BUFFER_SIZE,
                                      30,   // inside 阈值 (可调)
                                      5,    // none 阈值 (可调)
                                      20);  // outside 阈值 (可调)
        printf("%d\n", result);
        if (result == STATUS_ON_BED) {
            frame_count = 0;
			switch_to_range = true; // 判定为上床，切换至range fft模式
//			printf("Bed Status: On bed\n");
			int ret = mmw_ctrl_stop();
			if(ret){
				printf("close error: %d\n", ret);
			}
			mmw_point_cloud_deinit();
        } else if (result == STATUS_NONE) {
//			printf("Bed Status: No one or outside\n");
        }
        // 清空计数器，准备接收新的一批帧
        frame_count = 0;
		memset(real_data, 0, sizeof(real_data));
		memset(imag_data, 0, sizeof(imag_data));
		init_filter(); // 重置滤波器历史
    }
	return 0;
}

static void mmw_point_cloud_upload(const PointCloudBuffer_t *ptr_3d_pc, uint16_t pc_len) {
	float range_cm;
    float sin_y;
    float range_bin_size_cm;
    uint32_t range_mm, range_reol_mm;
    uint16_t range_fft_num, doppler_fft_num;

    Detection3D_Data detection_data = {0};
    Detection3D_State detection_state = {0};

    // 获取雷达参数
    mmw_range_get(&range_mm, &range_reol_mm);
    mmw_fft_num_get(&range_fft_num, &doppler_fft_num);
    range_bin_size_cm = (float)range_mm / range_fft_num / 10.f;

    if (pc_len == 0) {
        // 没有点时仍需调用 detection 回调，表示无检测目标
        detection_data.motion_points_num = 0;
        detection_data.motion_points = NULL;
        detection3d_cpuf_cb(&detection_data, &detection_state, NULL);
        return;
    }
	
	// 为 motion_points 分配内存
    mmw_process_mem_alloc((void**)&detection_data.motion_points, sizeof(PointCloud3D) * pc_len);
    if (!detection_data.motion_points) {
        return;
    }

    detection_data.motion_points_num = pc_len;
//	printf("Point cloud detected: %u points\n", detection_data.motion_points_num);
	
	// 填充点云数据
    for (uint16_t pc_idx = 0; pc_idx < pc_len; pc_idx++) {
        float azi = ptr_3d_pc->ptr_motion_point_cloud_data[pc_idx].azi_phase;
        float ele = ptr_3d_pc->ptr_motion_point_cloud_data[pc_idx].ele_phase;
        float range_idx = (float)ptr_3d_pc->ptr_motion_point_cloud_data[pc_idx].range_idx;

        range_cm = range_idx * range_bin_size_cm;
        sin_y = sqrtf(1 - azi * azi - ele * ele);

        float x, y, z;	
		float *buf_x = 0;
		float *ptr_X_Y_Z = 0;
		mmw_process_mem_alloc((void**)&ptr_X_Y_Z, sizeof(*ptr_X_Y_Z) * 3);
		if (!ptr_X_Y_Z) {
			return;
		}
		buf_x = ptr_X_Y_Z;

        mmw_point_cloud_trans_radar_coord_to_user_coord(
            range_cm * azi,   // radar x
            range_cm * sin_y, // radar y
            range_cm * ele,   // radar z
            buf_x        // 输出用户坐标系
        );
//		if(pc_idx<10){
//			printf("x:%d y:%d z:%d\n",(int)buf_x[0], (int)buf_x[1], (int)buf_x[2]);
//		}
        // 写入 cart 坐标形式
        detection_data.motion_points[pc_idx].cart.x = buf_x[0];
        detection_data.motion_points[pc_idx].cart.y = buf_x[1];
        detection_data.motion_points[pc_idx].cart.z = buf_x[2];
		mmw_process_mem_free((void**)&ptr_X_Y_Z);
    }
	
	// 其他字段置空
    detection_data.micro_points = NULL;
    detection_data.tracker_objs = NULL;
    detection_data.micro_points_num = 0;
    detection_data.tracker_objs_num = 0;

    // 状态信息，可按系统实际更新
//    detection_state.proc_frame_num++;
//    detection_state.proc_time_us = 0; // 处理时间（可测量后填）

    // 调用床状态检测或3D检测处理回调
    detection3d_cpuf_cb(&detection_data, &detection_state, NULL);

    // 释放内存
    mmw_process_mem_free((void**)&detection_data.motion_points);

}

/* ========== 主回调：由 mmw_ctrl 调用（每帧） ========== */
/* 该回调负责在点云模式下调用 mmw_point_cloud_process()，
 * 在 Range FFT 模式下调用 mmw_data_process()（处理 1D FFT）。
 */
static int mmw_ctrl_frame_cb_point(void *mmw_data, void *arg) {
	mmw_dsp_poweron();
	
	PointCloudBuffer_t *ptr_point_cloud_buffer;
	
	/* 处理点云数据 */
#if SW_CFAR_ENABLE
	ptr_point_cloud_buffer = mmw_point_cloud_process_sw_cfar();
#else
	ptr_point_cloud_buffer = mmw_point_cloud_process();
#endif

	/* 自动增益清除 */
	mmw_psic_auto_gain_clear();
	
	mmw_point_cloud_upload(ptr_point_cloud_buffer, ptr_point_cloud_buffer->point_cloud_num);
	mmw_process_mem_free((void**) &ptr_point_cloud_buffer->ptr_motion_point_cloud_data);
	mmw_process_mem_free((void**) &ptr_point_cloud_buffer);
    
    return 0;
}

static int mmw_ctrl_frame_cb_fft(void *mmw_data, void *arg) {
	mmw_dsp_poweron();
	mmw_data_process(mmw_data);
    
    return 0;
}

// 初始化点云配置的代码
void mmw_init_point(){
	int ret = mmw_mode_cfg(MMW_MIMO_2T4R, MMW_WORK_MODE_2DFFT);
	if (ret) {
		printf("mode cfg error! %d\n", ret);
	}
	ret = mmw_range_cfg(10000, 50);
	if (ret) {
		printf("range cfg error! %d\n", ret);
	}
	ret = mmw_velocity_cfg(1920, 60); //max=1.92m/s, resol=0.06m/s
	if (ret) {
		printf("velocity cfg error! %d\n", ret);
	}
	ret = mmw_frame_cfg(50, 0); //10Hz
	if (ret) {
		printf("frame cfg error! %d\n", ret);
	}

	ret = mmw_cfar_cfg(1, 80);
    if (ret) {
        printf("Cfar cfg error! %d\n", ret);
    }

	// 点云相关比较独特的配置
	ret = mmw_point_cloud_init();
	if (ret) {
		printf("mmw_point_cloud_init error! %d\n", ret);
	}

	ret = mmw_coordinate_config(MMW_COORDINATE_TYPE_CART);
	if(ret){
		printf("mmw_coordinate_init error! %d\n", ret);
	}

	ret = mmw_ctrl_callback_cfg(&mmw_ctrl_frame_cb_point, MMW_DATA_TYPE_2DFFT, NULL);
	if (ret) {
		printf("CPUS mmw_detection3d_callback_cfg error! %d\n", ret);
	}
	ret = mmw_ctrl_start();
	if (ret) {
		printf("mmw_detection3d_start error! %d\n", ret);
	}
}

// 初始化range fft配置的代码
void mmw_init_range(){
	int ret;
	ret = mmw_mode_cfg(MMW_MIMO_2T4R, MMW_WORK_MODE_1DFFT);
    // ret = mmw_mode_cfg(MMW_MIMO_1T1R, MMW_WORK_MODE_1DFFT); //demo version
    if(ret) {
        printk("mode cfg error! %d\n", ret);
    }
    ret = mmw_range_cfg(4000, 50); // 8m 0.1m
    if (ret) {
        printk("range cfg error! %d\n", ret);
    }
    ret = mmw_interval_cfg(606, 1);
    if(ret) {
        printk("Interval cfg error! %d\n", ret);
    }
    ret = mmw_frame_cfg(100, 0);
    if (ret) {
        printk("frame cfg error! %d\n", ret);
    }
    ret = mmw_ctrl_callback_cfg(&mmw_ctrl_frame_cb_fft, MMW_DATA_TYPE_1DFFT, NULL);
    if (ret) {
        printk("callback cfg error! %d\n", ret);
    }
	ret = mmw_ctrl_start();
	if (ret) {
		printk("sensor start error! %d\n", ret);
	}
}

int main(void) {
	
	if (LL_SYS_GetCurrentSysId()) {
        printf("current system : CPU-F [1]\n");
    } else {
        printf("current system : CPU-S [0]\n");
    }

    mmw_enable_dbg_uart(1);
#if (CONFIG_BOARD_PISC_MRS7241_P2528)
    mmw_set_dbg_uart_port(1, 0);
#else
    mmw_set_dbg_uart_port(1, 1);
#endif

    printf("mmw_ctrl_open\n");
	int ret = mmw_ctrl_open(true, false, true);
    if (ret) {
        printf("mmw open failed! %d\n", ret);
    } else {
        printf("mmw open success!\n");
    }

    printf("start init point cloud\n");
	mmw_init_point();

    printf("Bed status detection system started successfully!\n");
	while (1) {
        usleep(1000000); // 1s检测一次状态

        if (switch_to_point) {
            switch_to_point = false;
            ret = mmw_ctrl_stop();
            if (ret) {
                printf("close error: %d\n", ret);
            }
			// k_sleep(K_MSEC(200));   // 加延迟，避免频繁启停死机
            mmw_point_cloud_deinit(); 
			mmw_init_point();
//			printf("Switch to Point Cloud Mode!\n");
            frame_count = 0;
        } else if (switch_to_range) {
            switch_to_range = false;
			ret = mmw_ctrl_stop();
            if (ret) {
                printf("close error: %d\n", ret);
            }
            // k_sleep(K_MSEC(200));   // 同样加延迟
			mmw_point_cloud_deinit();
            mmw_init_range();
//			printf("Switch to Range FFT Mode!\n");
        }
    }

    return 0;
}

/*
 ******************************************************************************
 * (C) COPYRIGHT POSSUMIC TECHNOLOGY
 * END OF FILE
 */
