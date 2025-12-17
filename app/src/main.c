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
//Complex16_RealImag buffer[RANGE_BIN_NUM + 1] __aligned(4);
Complex16_RealImag buffer[RANGE_BIN_NUM + 1]={0};
int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM] = {0};
int16_t imag_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM] = {0};

// 用于choose_bin_final
float phase_tmp[FRAME_BUFFER_SIZE];
float phase_sequences[RANGE_BIN_NUM][FRAME_BUFFER_SIZE];
float energies[RANGE_BIN_NUM];
float auto_vals[RANGE_BIN_NUM];
float scores[RANGE_BIN_NUM];
float ac_pos[FRAME_BUFFER_SIZE];
int   peaks[32];

// 用于estimate_respiration_rate
float phase[FRAME_BUFFER_SIZE];
float phase_filtered[FRAME_BUFFER_SIZE];
int   phase_filtered_peaks[32];
float phase_filtered_bpm;

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
#define M_PI 3.14159265358979323846f
#endif

float unwrap_phase(float prev_unwrapped, float cur_raw)
{
    float two_pi = 2.0f * M_PI;
    float prev_raw = fmodf(prev_unwrapped + M_PI, two_pi);
    if (prev_raw < 0)
        prev_raw += two_pi;
    prev_raw -= M_PI;  

    float diff = cur_raw - prev_raw;

    float diff_mod = fmodf(diff + M_PI, two_pi);
    if (diff_mod < 0)
        diff_mod += two_pi;
    diff_mod -= M_PI;

    if (diff_mod == -M_PI && diff > 0)
        diff_mod = M_PI;

    return prev_unwrapped + diff_mod;
}

float sig_buffer[FRAME_BUFFER_SIZE + 200]={0};   
float tmp_buffer[FRAME_BUFFER_SIZE + 200]={0};

// 反射填充函数
void reflect_pad_numpy(const float* signal, float* padded, int n, int pad_width)
{
    int padded_len = n + 2 * pad_width;
    if (n <= 0 || pad_width <= 0) return;

    // 中间复制原信号
    for (int i = 0; i < n; i++) {
        padded[i + pad_width] = signal[i];
    }

    // 左填充：反射，不重复边界元素，即 padded[pad-1] = signal[1]
    // 如果信号太短（n<=2），退化为重复边界
    if (n >= 2) {
        for (int i = 0; i < pad_width; i++) {
            int src_idx = 1 + i;
            if (src_idx >= n) src_idx = n - 1; // 保护
            padded[pad_width - 1 - i] = signal[src_idx];
        }
    } else {
        // n == 1 时直接复制第0元素
        for (int i = 0; i < pad_width; i++) {
            padded[pad_width - 1 - i] = signal[0];
        }
    }

    // 右填充：signal[n-2], signal[n-3]...
    if (n >= 2) {
        for (int i = 0; i < pad_width; i++) {
            int src_idx = n - 2 - i;
            if (src_idx < 0) src_idx = 0;
            padded[pad_width + n + i] = signal[src_idx];
        }
    } else {
        for (int i = 0; i < pad_width; i++) {
            padded[pad_width + n + i] = signal[0];
        }
    }
}

// 使用卷积计算移动平均
void moving_average_conv(const float* sig, float* out, int n, int window_size)
{
    if (n <= 0) return;
    if (window_size <= 1) {
        // 直接拷贝
        for (int i = 0; i < n; i++) out[i] = sig[i];
        return;
    }

    int pad_width = window_size / 2;
    // 如果窗口超过信号长度，退化为整个信号均值（匹配 Python 的 behavior）
    if (window_size >= n) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += sig[i];
        float mean = s / (float)n;
        for (int i = 0; i < n; i++) out[i] = mean;
        return;
    }

    // 反射填充到 tmp_buffer（要求 tmp_buffer 大小 >= n + 2*pad_width）
    reflect_pad_numpy(sig, tmp_buffer, n, pad_width);

    // 初始窗口求和（tmp_buffer 前 pad_width...）
    float window_sum = 0.0f;
    for (int i = 0; i < window_size; i++) {
        window_sum += tmp_buffer[i];
    }

    // 卷积 - 输出长度 n，对应 tmp_buffer 索引从 0..(n-1)
    for (int i = 0; i < n; i++) {
        out[i] = window_sum / (float)window_size;
        // 滑动窗口：移除 tmp_buffer[i]，加入 tmp_buffer[i + window_size]
        window_sum -= tmp_buffer[i];
        window_sum += tmp_buffer[i + window_size];
    }
}

void remove_baseline_drift(float* signal, int n)
{
    const float fs = FS;      // 20.0f
    const float win_len = 3.0f;

    int window_size = (int)(fs * win_len);  // 60
    // 保证为奇数以便中心对称（与 Python 的 window//2 行为一致并更稳定）
    if (window_size % 2 == 0) window_size++;

    // 防止窗口过大
    if (window_size > n) {
        // 退化为 n(或最接近的奇数)
        window_size = (n % 2 == 0) ? n - 1 : n;
        if (window_size < 3) window_size = 3;
    }

    // tmp_buffer 作为 padded + 中间结果（预分配）
    moving_average_conv(signal, tmp_buffer, n, window_size);

    // 原信号减去 baseline
    for (int i = 0; i < n; i++) {
        signal[i] -= tmp_buffer[i];
    }
}

// butterworth带通滤波
#define FILTER_ORDER 4
float b[FILTER_ORDER + 1] = {
	0.00554272f, 
	0.0000000f, 
	-0.01108543f, 
	0.00000000f, 
	0.00554272f
};
float a[FILTER_ORDER + 1] = {
	1.00000000f, 
	-3.76742617f, 
	5.33686791f, 
	-3.37021291f, 
	0.80080265f 
};

float x_history[FILTER_ORDER] = {0};
float y_history[FILTER_ORDER] = {0};
int filter_initialized = 0;

// 初始化滤波器（重置历史数据）
void init_filter(void) {
    memset(x_history, 0, FILTER_ORDER * sizeof(float));
    memset(y_history, 0, FILTER_ORDER * sizeof(float));
    filter_initialized = 1;
}

// 对单个样本进行滤波
float filter_sample(float x_n) {
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
void filter_array(float *x, float *y, int n) {
	init_filter();
    
    for (int i = 0; i < n; i++) {
        y[i] = filter_sample(x[i]);
    }
}

#define MIN_PEAK_DISTANCE (int)(FS * 2.0)  // 峰间最小距离（0.6秒）
//
// 检测峰值
int find_peaks(const float *signal, int length, int *peaks, int max_peaks) {
    int count = 0;
    int last_peak = -MIN_PEAK_DISTANCE;

    for (int i = 1; i < length - 1; i++) {
        // 简单的局部极大值检测
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
            // 距离约束：相邻峰至少间隔 MIN_PEAK_DISTANCE
            if (i - last_peak >= MIN_PEAK_DISTANCE) {
                if (count < max_peaks) {
                    peaks[count++] = i;
                    last_peak = i;
                }
            }
        }
    }
    return count;  // 返回检测到的峰数量
}

// 呼吸率计算
float compute_breath_rate(const float *signal, int length, int num_peaks, int *peaks) {

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

float compute_mean(float* phase)
{
	float mean = 0.0f;
	for (int t = 0; t < FRAME_BUFFER_SIZE; t++) mean += phase[t];
	mean /= (float)FRAME_BUFFER_SIZE;
	return mean;
}

int choose_bin(int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM],
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
		remove_baseline_drift(phase_tmp, FRAME_BUFFER_SIZE);
	
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
		
		// 计算自相关
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
		
		int peak_count = find_peaks(ac_pos, N, peaks, 32);
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
	return best_bin;
}

// 主函数：计算呼吸率 (bpm)
float estimate_respiration_rate(int16_t real_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM],
                                int16_t imag_data[FRAME_BUFFER_SIZE][RANGE_BIN_NUM])
{
	
	// 1. 选择bin（只做预处理，不做完整滤波）
    int best_bin = choose_bin(real_data, imag_data);
//	printf("best_bin:%d\n",best_bin);
	
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
	remove_baseline_drift(phase, FRAME_BUFFER_SIZE);
	
	// 4. 带通滤波
	init_filter();
	filter_array(phase, phase_filtered, FRAME_BUFFER_SIZE);
	
    // 5. 计算呼吸率
	int num_peaks = find_peaks(phase_filtered, FRAME_BUFFER_SIZE, phase_filtered_peaks, 32);
    phase_filtered_bpm = compute_breath_rate(phase_filtered, FRAME_BUFFER_SIZE, num_peaks, phase_filtered_peaks);
//	printf("bpm:%.2f\n",phase_filtered_bpm);
	
    return phase_filtered_bpm;
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
        ret = mmw_fft_data(buffer, RANGE_BIN_NUM+1, tx, rx, 16);
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
//			printf("Range FFT Data Collection Full!\n");
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
    ret = mmw_frame_cfg(50, 0);
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
